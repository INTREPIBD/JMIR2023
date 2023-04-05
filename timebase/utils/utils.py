import os
import re
import csv
import copy
import subprocess
import typing as t
import numpy as np
from glob import glob
import tensorflow as tf

from timebase.utils import yaml


def save_checkpoint(
    args, checkpoint: tf.train.Checkpoint, epoch: int, filename: str = None
):
    if filename is None:
        filename = os.path.join(args.checkpoint_dir, f"epoch-{epoch:03d}")
    else:
        filename = os.path.join(args.checkpoint_dir, filename)
    path = checkpoint.write(filename)
    if args.verbose == 2:
        print(f"saved checkpoint to {filename}\n")
    return path


def load_checkpoint(args, checkpoint: tf.train.Checkpoint, force: bool = False):
    """
    Load the best checkpoint or the latest checkpoint from args.checkpoint_dir
    if available, and return the epoch number of that checkpoint.
    Args:
      args
      checkpoint: tf.train.Checkpoint, TensorFlow Checkpoint object
      force: bool, raise an error if no checkpoint is found.
    Returns:
      epoch: int, the epoch number of the loaded checkpoint, 0 otherwise.
    """
    epoch, ckpt_filename = 0, None
    # load best model if exists, otherwise load the latest model if exists.
    best_model_yaml = os.path.join(args.checkpoint_dir, "best_model.yaml")
    if os.path.exists(best_model_yaml):
        best_model_info = yaml.load(best_model_yaml)
        epoch = best_model_info["epoch"]
        ckpt_filename = os.path.join(args.checkpoint_dir, best_model_info["path"])
    else:
        checkpoints = sorted(glob(os.path.join(args.checkpoint_dir, "*.index")))
        if checkpoints:
            ckpt_filename = checkpoints[-1].replace(".index", "")
    if force and not ckpt_filename:
        raise FileNotFoundError(f"no checkpoint found in {args.output_dir}.")
    if ckpt_filename:
        status = checkpoint.restore(ckpt_filename)
        status.expect_partial()
        if epoch == 0:
            match = re.match(r".+epoch-(\d{3})", ckpt_filename)
            epoch = int(match.groups()[0])
        if args.verbose:
            print(f"loaded checkpoint from {ckpt_filename}\n")
    return epoch


def update_dict(target: t.Dict, source: t.Dict, replace: bool = False):
    """add or update items in source to target"""
    for key, value in source.items():
        if replace:
            target[key] = value
        else:
            if key not in target:
                target[key] = []
            target[key].append(value)


def check_output(command: list):
    """Execute command in subprocess and return output as string"""
    return subprocess.check_output(command).strip().decode()


def save_args(args):
    """Save args object as dictionary to output_dir/args.yaml"""
    arguments = copy.deepcopy(args.__dict__)
    # remove session2class as yaml does not support tuple
    arguments.pop("session2class", None)
    arguments["git_hash"] = check_output(["git", "describe", "--always"])
    arguments["hostname"] = check_output(["hostname"])
    yaml.save(filename=os.path.join(args.output_dir, "args.yaml"), data=arguments)


def load_args(args, experiment):
    """Load args from output_dir/args.yaml"""
    filename = os.path.join(
        [f for f in glob(os.path.join(experiment, "*")) if f.endswith(args.algorithm)][
            0
        ],
        "args.yaml",
    )
    assert os.path.exists(filename)
    arguments = yaml.load(filename=filename)
    for key, value in arguments.items():
        if key not in [
            "class2name",
            "class2session",
            "session2class",
            "train_steps",
            "val_steps",
            "test_steps",
            "ds_info",
        ]:
            setattr(args, key, value)


def load_args_oos(args):
    """Load args from output_dir/args.yaml"""
    filename = os.path.join(args.path2model, "args.yaml")
    assert os.path.exists(filename)
    arguments = yaml.load(filename=filename)
    for key, value in arguments.items():
        if not hasattr(args, key) and key not in [
            "class2name",
            "class2session",
            "session2class",
            "train_steps",
            "val_steps",
            "test_steps",
            "ds_info",
            "output_dir"
        ]:
            setattr(args, key, value)


def accuracy(cm: np.ndarray):
    """Compute accuracy given Numpy array confusion matrix cm. Returns a
    floating point value"""
    return np.sum(np.diagonal(cm)) / np.sum(cm)


def write_csv(output_dir, content: list):
    with open(os.path.join(output_dir, "results.csv"), "a") as file:
        writer = csv.writer(file)
        writer.writerow(content)
