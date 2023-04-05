import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import kendall_w as kw

from timebase.data.reader import scramble_test_ds, get_datasets
from timebase.utils import utils, tensorboard, yaml
from timebase.models.classifiers.registry import get_model

features = ["ACC", "BVP", "EDA", "HR", "TEMP"]


def cross_entropy(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.reduce_mean(
        tf.losses.sparse_categorical_crossentropy(
            y_true=y_true, y_pred=y_pred, from_logits=False
        )
    )


def accuracy(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.reduce_mean(
        tf.metrics.sparse_categorical_accuracy(y_true=y_true, y_pred=y_pred)
    )


@tf.function
def test_step(x, y, model):
    y_pred = model(x, training=False)
    return {
        "loss": cross_entropy(y_true=y, y_pred=y_pred),
        "accuracy": accuracy(y_true=y, y_pred=y_pred),
    }


def test(args, ds, model):
    results = {}
    for x, y in tqdm(ds, desc="Test", total=args.test_steps,
                     disable=args.verbose == 0):
        result = test_step(x, y, model=model)
        utils.update_dict(target=results, source=result)
    for k, v in results.items():
        results[k] = tf.reduce_mean(v).numpy()
    print(
        f'Test\t\tloss: {results["loss"]:.04f}\t'
        f'accuracy: {results["accuracy"] * 100:.02f}%'
    )
    return results


def plot_feature_importance(args, df: pd.DataFrame,
                            features_importance: np.ndarray):
    rankings = np.empty_like(features_importance.T).astype("int")
    for i in range(features_importance.T.shape[1]):
        order = features_importance.T[:, i].argsort()
        ranks = order.argsort()
        rankings[:, i] = ranks
    # https://pypi.org/project/kendall-w/
    W = kw.compute_w(rankings.tolist())

    df.loc[:, "Value"] = round(df.loc[:, "Value"] * 100, 3)
    fig, ax = plt.subplots(figsize=(15, len(df.Channel.unique()) * 1.5),
                           dpi=args.dpi)
    ax = sns.barplot(
        x="Channel",
        y="Value",
        hue="Experiment",
        data=df,
        order=df.Channel.unique().tolist(),
    )

    ax.set_xlabel("Sensor")
    ax.set_ylabel("Permutation Importance (Accuracy)")
    ax.set_title(f"Kendall W = {round(W, 3)}")
    fig.savefig(os.path.join(args.output_dir,  f"permutation_importance"
                                               f".{args.format}"))

    tensorboard.save_figure(
        fig,
        filename=os.path.join(
            args.experiment_dir,
            f"{args.algorithm}_permutation_importance.{args.format}",
        ),
        close=False,
    )
    plt.close(fig)
    summary = {
        "Channel": df.Channel.unique().tolist(),
        "Mean": np.round(np.mean(features_importance, axis=0) * 100, 3),
        "SD": np.round(np.std(features_importance, axis=0) * 100, 3),
    }

    if args.verbose:
        print(pd.DataFrame(summary))

    yaml.save(
        filename=os.path.join(
            args.experiment_dir, f"{args.algorithm}_permutation_importance.yaml"
        ),
        data=summary,
    )


def main(args):
    assert os.path.isdir(args.experiment_dir)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    experiments = [
        x
        for x in sorted(glob(os.path.join(args.experiment_dir, "*")))
        if os.path.isdir(x)
    ]
    if args.verbose:
        print(f"{len(experiments)} found under {args.experiment_dir}")
    features_importance = np.empty(
        shape=(
            len(experiments),
            len(features),
        )
    )

    for i, experiment in enumerate(experiments):
        utils.load_args(args, experiment)
        model = get_model(args)
        checkpoint = tf.train.Checkpoint(model=model)
        utils.load_checkpoint(
            args,
            checkpoint=checkpoint,
            force=True,
            epoch=list(args.val_record.keys())[0],
        )
        _, _, test_ds = get_datasets(args)
        test_results = test(args, ds=test_ds, model=model)
        baseline = test_results["accuracy"]

        x_test, y_test = [], []
        for x, y in test_ds:
            x_test.append(x.numpy())
            y_test.append(y.numpy())
        x_test, y_test = np.concatenate(x_test), np.concatenate(y_test)
        permutation_importance_under_task = []
        for f in features:
            channel_idx = [
                idx
                for idx, channel in enumerate(
                    [c.replace("$", "") for c in args.ds_info["channel_names"]]
                )
                if channel.startswith(f)
            ]
            test_ds_permuted = scramble_test_ds(
                args, x_test=x_test, y_test=y_test, to_permute=channel_idx
            )
            test_results_permutation = test(args,
                                            ds=test_ds_permuted,
                                            model=model)
            permutation_importance = 1 - (
                test_results_permutation["accuracy"] / baseline
            )
            permutation_importance_under_task.append(permutation_importance)

        features_importance[i] = permutation_importance_under_task
        print(len(permutation_importance_under_task))
        print(f"{experiment}")

    data = {
        "Channel": features * len(experiments),
        "Experiment": np.repeat(experiments, len(features)),
        "Value": features_importance.flatten(),
    }
    df = pd.DataFrame(data)
    plot_feature_importance(args, df, features_importance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # configuration
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--context", type=str, default="feature importance")

    # matplotlib
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--format", type=str, default="pdf",
                        choices=["pdf", "png"])
    parser.add_argument("--save_plots", action="store_true")

    # misc
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    main(parser.parse_args())
