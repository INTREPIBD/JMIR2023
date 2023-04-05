import os
import shutil
import argparse
from tqdm import tqdm
from time import time
import tensorflow as tf
import tensorflow_addons as tfa

from timebase.data.reader import get_datasets
from timebase.utils.early_stopping import EarlyStopping
from timebase.models.classifiers.registry import get_model
from timebase.utils import tensorboard, utils, yaml, metrics, plots
from timebase.utils.optimizer import Optimizer


@tf.function
def train_step(x, y, model: tf.keras.Model, optimizer: Optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = metrics.cross_entropy(y_true=y, y_pred=y_pred)
        scaled_loss = optimizer.get_scaled_loss(loss)
    optimizer.minimize(loss=scaled_loss, tape=tape)
    return {"loss": loss, "accuracy": metrics.accuracy(y_true=y, y_pred=y_pred)}


def train(
    args,
    ds: tf.data.Dataset,
    model: tf.keras.Model,
    optimizer: Optimizer,
    summary: tensorboard.Summary,
    epoch: int,
):
    results = {}
    for x, y in tqdm(
        ds, desc="Train", total=args.train_steps, disable=args.verbose == 0
    ):
        result = train_step(x, y, model=model, optimizer=optimizer)
        utils.update_dict(target=results, source=result)
    for k, v in results.items():
        results[k] = tf.reduce_mean(v).numpy()
        summary.scalar(k, value=results[k], step=epoch, mode=0)
    return results


@tf.function
def validation_step(x, y, model: tf.keras.Model):
    y_pred = model(x, training=False)
    return {
        "loss": metrics.cross_entropy(y_true=y, y_pred=y_pred),
        "accuracy": metrics.accuracy(y_true=y, y_pred=y_pred),
    }


def validate(
    args,
    ds: tf.data.Dataset,
    model: tf.keras.Model,
    summary: tensorboard.Summary,
    epoch: int,
):
    results = {}
    for x, y in tqdm(
        ds, desc="Validation", total=args.val_steps, disable=args.verbose == 0
    ):
        result = validation_step(x, y, model=model)
        utils.update_dict(target=results, source=result)
    for k, v in results.items():
        results[k] = tf.reduce_mean(v).numpy()
        summary.scalar(k, value=results[k], step=epoch, mode=1)
    return results


def test(
    args,
    ds: tf.data.Dataset,
    model: tf.keras.Model,
    summary: tensorboard.Summary,
    epoch: int,
):
    results = {}
    for x, y in tqdm(ds, desc="Test", total=args.test_steps, disable=args.verbose == 0):
        result = validation_step(x, y, model=model)
        utils.update_dict(target=results, source=result)
    for k, v in results.items():
        results[k] = tf.reduce_mean(v).numpy()
        summary.scalar(k, value=results[k], step=epoch, mode=2)
    if args.verbose:
        print(
            f'Test\t\tloss: {results["loss"]:.04f}\t'
            f'accuracy: {results["accuracy"] * 100:.02f}%'
        )
    return results


def main(args):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)

    if args.clear_output_dir and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.mixed_precision:
        if args.verbose:
            print(f"Enable mixed precision training.")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    train_ds, val_ds, test_ds = get_datasets(args)

    summary = tensorboard.Summary(args)

    model = get_model(args, summary)
    optimizer = Optimizer(args, model=model)

    args.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer.optimizer)

    early_stopping = EarlyStopping(args, model=model, checkpoint=checkpoint)
    epoch = utils.load_checkpoint(args, checkpoint=checkpoint)

    utils.save_args(args)
    # plots.model_res(args=args, ds=val_ds, model=model, summary=summary, epoch=epoch)
    results = {}
    while (epoch := epoch + 1) < args.epochs + 1:
        if args.verbose:
            print(f"Epoch {epoch:03d}/{args.epochs:03d}")

        start = time()
        train_results = train(
            args,
            ds=train_ds,
            model=model,
            optimizer=optimizer,
            summary=summary,
            epoch=epoch,
        )
        val_results = validate(
            args, ds=val_ds, model=model, summary=summary, epoch=epoch
        )
        elapse = time() - start

        summary.scalar("elapse", value=elapse, step=epoch, mode=0)
        if args.verbose:
            print(
                f'Train\t\tloss: {train_results["loss"]:.04f}\t'
                f'accuracy: {train_results["accuracy"]*100:.02f}%\n'
                f'Validation\tloss: {val_results["loss"]:.04f}\t'
                f'accuracy: {val_results["accuracy"]*100:.02f}%\n'
                f"Elapse: {elapse:.02f}s\n"
            )

        results.update({"train": train_results, "validation": val_results})
        if early_stopping.monitor(loss=val_results["loss"], epoch=epoch):
            break
        if epoch % 10 == 0 or epoch == args.epochs:
            plots.model_res(
                args=args, ds=val_ds, model=model, summary=summary, epoch=epoch
            )
    early_stopping.restore()

    test_results = test(
        args, ds=test_ds, model=model, summary=summary, epoch=early_stopping.best_epoch
    )
    results.update({"test": test_results})
    plots.model_res(
        args=args,
        ds=val_ds,
        model=model,
        summary=summary,
        epoch=early_stopping.best_epoch,
        mode=2,
        results=results["test"],
    )
    yaml.save(os.path.join(args.output_dir, "results.yaml"), data=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training configuration
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mixed_precision", action="store_true")

    # dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/raw_data",
        help="path to directory with raw data in zip files",
    )
    parser.add_argument(
        "--classification_mode",
        type=int,
        default=0,
        choices=[0],
        help="classification mode: 0) classify session ID",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to .yaml file that contains classification labels",
    )
    parser.add_argument(
        "--downsampling",
        type=str,
        default="average",
        choices=["average", "max"],
        help="downsampling method to use",
    )
    parser.add_argument(
        "--time_alignment",
        type=int,
        default=1,
        choices=[1, 2, 4, 8, 16, 32, 64],
        help="number of samples per second (Hz) for time-alignment",
    )
    parser.add_argument(
        "--norm_mode",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="normalize features: "
        "0) no normalization "
        "1) normalize features by same scale"
        "2) normalize features per session",
    )
    parser.add_argument(
        "--padding_mode",
        type=str,
        default="average",
        choices=["zero", "last", "average", "median"],
        help="padding mode for channels samples at a lower frequency",
    )
    parser.add_argument(
        "--filter_mode",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="filtering mode:"
        "0 - no filtering"
        "1 - filter recordings where all channels are zeros for more than 10s"
        "2 - Kleckner et al. 2018 - https://pubmed.ncbi.nlm.nih.gov/28976309/",
    )
    parser.add_argument(
        "--ibi_interpolation",
        type=str,
        default="quadratic",
        choices=["linear", "quadratic"],
        help="interpolation method to use in IBI channel",
    )
    parser.add_argument(
        "--hrv_features",
        nargs="+",
        default=[],
        help="choose which HRV features should be extracted from IBI",
    )
    parser.add_argument(
        "--hrv_length",
        type=int,
        default=60 * 5,
        help="window length for computing HRV from IBI",
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=32,
        help="segmentation window length in seconds",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=20,
        help="number of segments from each session for testing",
    )

    # model configuration
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--num_units", type=int, default=128)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    # RNNs configuration
    parser.add_argument(
        "--r_dropout", type=float, default=0.0, help="Recurrent dropout in RNNs."
    )

    # Transformer configuration
    parser.add_argument("--num_encoders", type=int, default=4)
    parser.add_argument("--head_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=4)
    parser.add_argument(
        "--t_dropout",
        type=float,
        default=0.25,
        help="Dropout rate for the Transformer encoder.",
    )

    # matplotlib
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument(
        "--format", type=str, default="pdf", choices=["pdf", "png", "svg"]
    )
    parser.add_argument("--dpi", type=int, default=120)

    # misc
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--clear_output_dir", action="store_true")

    params = parser.parse_args()
    main(params)
