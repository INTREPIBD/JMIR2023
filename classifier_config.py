import os
import argparse
from tqdm import tqdm
import tensorflow as tf

from timebase.data.reader import get_datasets
from timebase.models.classifiers.registry import get_model
from timebase.utils import tensorboard, utils, yaml, metrics, plots


@tf.function
def test_step(x, y, model):
    y_pred = model(x, training=False)
    return {
        "loss": metrics.cross_entropy(y_true=y, y_pred=y_pred),
        "accuracy": metrics.accuracy(y_true=y, y_pred=y_pred),
    }


def test(args, ds, model, summary, epoch: int):
    results = {}
    for x, y in tqdm(ds, desc="Test", total=args.test_steps,
                     disable=args.verbose == 0):
        result = test_step(x, y, model=model)
        utils.update_dict(target=results, source=result)
    for k, v in results.items():
        results[k] = tf.reduce_mean(v).numpy()
        summary.scalar(k, value=results[k], step=epoch, mode=0)
    print(
        f'Test\t\tloss: {results["loss"]:.04f}\t'
        f'accuracy: {results["accuracy"] * 100:.02f}%'
    )
    return results


def main(args):
    assert os.path.isdir(args.path2model)

    utils.load_args_oos(args)

    tf.keras.utils.set_random_seed(args.seed)

    if args.verbose:
        print(f"load configuration {args.config}")

    train_ds, val_ds, test_ds = get_datasets(args)

    args.output_dir = os.path.join(
        args.path2model,
        f"{os.path.splitext(os.path.basename(args.config))[0]}",
    )
    summary = tensorboard.Summary(args)

    model = get_model(args, summary)

    checkpoint = tf.train.Checkpoint(model=model)
    epoch = utils.load_checkpoint(args, checkpoint=checkpoint, force=True)

    test_results = test(args, ds=test_ds, model=model, summary=summary,
                        epoch=epoch)
    plots.model_res(
        args=args,
        ds=test_ds,
        model=model,
        summary=summary,
        epoch=epoch,
        mode=2,
        results=test_results,
    )
    yaml.save(
        os.path.join(args.output_dir, "result.yaml"),
        data=test_results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # out-of-sample test configuration
    parser.add_argument("--path2model", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to .yaml file that contains classification labels",
    )

    main(parser.parse_args())
