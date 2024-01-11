import argparse
import json
import logging
import math
from pathlib import Path
import time

import evaluate
import torch
import os
from datasets import load_dataset
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, SchedulerType, get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.distributed_c10d import ReduceOp

import torch.distributed as dist
import smdistributed.dataparallel.torch.torch_smddp
import smppy
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser(description="Fine-tune a Transformers model on an image classification dataset")

        
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cifar10",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset)."
        ),
    )
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"], help="A folder containing the training data.")
    parser.add_argument("--validation_dir", type=str, default=None, help="A folder containing the validation data.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.15,
        help="Percent to split off of train for validation",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="google/vit-base-patch16-224-in21k",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_false",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    args = parser.parse_args()
    # Sanity checks
    if args.dataset_name is None and args.train_dir is None and args.validation_dir is None:
        raise ValueError("Need either a dataset name or a training/validation folder.")

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])

    print(f"Local rank {args.local_rank} Global Rank {args.rank} World Size {args.world_size}")
    return args


def main():
    args = parse_args()


    ################################################################
    torch.distributed.init_process_group(
                "nccl"
            )

    ##########################

    LOG_DIR="/opt/ml/output/tensorboard"
    writer= SummaryWriter(log_dir=LOG_DIR)

    ###############################
    SMProf = smppy.SMProfiler.instance()
    config = smppy.Config()
    config.profiler = {
        "EnableCuda": "1",
    }
    SMProf.configure(config)
    SMProf.start_profiling()
    ######################################
    # If passed along, set the training seed now.
    if args.seed is not None:
        torch.manual_seed(args.seed)

    #####################################

    dataset = load_from_disk(args.train_dir)
    # If we don't have a validation split, split off a percentage of train as validation.
    args.train_val_split = None if "validation" in dataset.keys() else args.train_val_split
    if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = dataset["train"].features["label"].names
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    # Load pretrained model and image processor
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(labels),
        i2label=id2label,
        label2id=label2id,
        finetuning_task="image-classification",
    )
    image_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )

    # Preprocessing the datasets

    # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    #with accelerator.main_process_first():
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)
    if args.max_eval_samples is not None:
        dataset["validation"] = dataset["validation"].shuffle(seed=args.seed).select(range(args.max_eval_samples))
    # Set the validation transforms
    eval_dataset = dataset["validation"].with_transform(preprocess_val)

    # DataLoaders creation:
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}


    train_sampler = torch.utils.data.DistributedSampler(
                train_dataset,
                shuffle=True,
                seed=args.seed,
                rank=args.rank,
                num_replicas=args.world_size,
                drop_last=True,
            )
    
    eval_sampler = torch.utils.data.DistributedSampler(
                eval_dataset,
                shuffle=True,
                seed=args.seed,
                rank=args.rank,
                num_replicas=args.world_size,
                drop_last=True,
            )
    
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_training_steps = args.num_train_epochs * len(train_dataloader)

    eval_dataloader = DataLoader(eval_dataset,sampler=eval_sampler, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=num_training_steps * args.gradient_accumulation_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    #num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    #args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Get the metric function
    metric = evaluate.load("accuracy")
    clf_metrics = evaluate.combine([
        evaluate.load("accuracy",average="weighted"),
        evaluate.load("f1",average="weighted"),
        evaluate.load("precision", average="weighted"),
        evaluate.load("recall", average="weighted")
        ])
    
    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size
    if args.rank==0:
        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(num_training_steps),disable=not (args.local_rank == 0))
    completed_steps = 0
    starting_epoch = 0

    device = torch.device(f"cuda:{args.local_rank}")
    
    model = DDP(model.to(device))
    torch.cuda.set_device(args.local_rank)
    model.cuda(args.local_rank)

    train_step_count = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()

        total_loss = 0
        for  step,batch in enumerate(train_dataloader):
            with smppy.annotate("step_"+str(step)):
                train_start = time.perf_counter()

                batch = {k: v.to(device) for k, v, in batch.items()}
                with smppy.annotate("Forward"):
                    outputs = model(**batch)
                with smppy.annotate("Loss"):
                    loss = outputs.loss
                    # We keep track of the loss at each epoch
                torch.distributed.all_reduce(loss, ReduceOp.SUM)
                loss = loss / args.world_size
                writer.add_scalar("Loss/train", loss, step)
                total_loss += loss.detach().float()
                train_bp_start = time.perf_counter()
                writer.add_scalar("FP latency/train", np.array(train_bp_start - train_start), step)
                with smppy.annotate("Backward"):
                    loss.backward()
                writer.add_scalar("BP latency/train",  np.array(time.perf_counter() - train_bp_start), step)
                train_optim_start = time.perf_counter()
                with smppy.annotate("Optimizer"):
                    optimizer.step() #gather gradient updates from all cores and apply them
                    lr_scheduler.step()
                    optimizer.zero_grad()
                writer.add_scalar("Optimizer latency/train",  np.array(time.perf_counter() - train_optim_start), step)
                writer.add_scalar("total step latency/train",  np.array(time.perf_counter() - train_start), step)
                train_step_count += 1

                progress_bar.update(1)
                completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break
                  
                model.eval()
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        batch = {k: v.to(device) for k, v, in batch.items()}
                        outputs = model(**batch)
                        eval_loss = outputs.loss
                    predictions = outputs.logits.argmax(dim=-1)
                    references = batch["labels"]
                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )

                eval_metric = metric.compute()
           
                writer.add_scalar("Loss/Eval", eval_loss, epoch)
                writer.add_scalar("Accuracy/Eval", torch.tensor(eval_metric["accuracy"]), epoch)
        
        if args.rank == 0:
            print(
                        "Epoch {}, Train Loss {:0.4f}".format(epoch, loss.detach().to("cpu"))
                )  
            print(f"epoch {epoch}: {eval_metric}")
            print(f"epoch {epoch}: eval loss {eval_loss}")

    SMProf.stop_profiling()

    if args.output_dir is not None and args.rank == 0:     
        image_processor.save_pretrained(args.output_dir)
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f)

    dist.barrier()

if __name__ == "__main__":
    main()
