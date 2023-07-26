import os
import numpy as np
from scipy import stats
import typing as t
from math import ceil
import tensorflow as tf

from timebase.utils import yaml
from timebase.data import preprocessing, utils

AUTOTUNE = tf.data.AUTOTUNE


def dry_run(datasets: t.List[tf.data.Dataset]):
    count = 0
    for ds in datasets:
        for _ in ds:
            count += 1


def set_classification_mode(args):
    """
    Set the corresponding options for the specified classification mode.

    args.classification_mode
      0: classify session IDs
    """
    if args.classification_mode == 0:
        config = yaml.load(args.config)
        args.session2class = {
            tuple(v["id"]) if type(v["id"]) == list else (v["id"],): k
            for k, v in config.items()
        }
        args.class2name = {k: v["name"] for k, v in config.items()}
        args.class2session = {v: k for k, v in args.session2class.items()}
        args.num_classes = len(args.session2class)
    else:
        raise NotImplementedError(
            f"classification mode {args.classification_mode} not implemented."
        )


def normalize_features(args, data: t.Dict[str, np.ndarray]):
    """Normalize features to [0, 1] according to args.norm_mode:
    - 0: no normalization
    - 1: normalize features from all sessions using the overall min and max value
    - 2: normalize features from each session using the session's min and max value
    """
    session_ids = []
    for s in args.session2class.keys():
        session_ids.extend(list(s))

    if args.norm_mode == 0:
        pass
    elif args.norm_mode == 1:
        s_min = np.array([args.ds_info["sessions_info"][s]["min"] for s in session_ids])
        s_max = np.array([args.ds_info["sessions_info"][s]["max"] for s in session_ids])
        ds_min, ds_max = np.min(s_min, axis=0), np.max(s_max, axis=0)
        for k in ["x_train", "x_val", "x_test"]:
            data[k] = utils.normalize(data[k], x_min=ds_min, x_max=ds_max)
    elif args.norm_mode == 2:
        for k1, k2 in [
            ("x_train", "y_train"),
            ("x_val", "y_val"),
            ("x_test", "y_test"),
        ]:
            for s in session_ids:
                x, y = data[k1], data[k2]
                indexes = np.where(y[:, 0] == s)[0]
                s_min = args.ds_info["sessions_info"][s]["min"]
                s_max = args.ds_info["sessions_info"][s]["max"]
                data[k1][indexes] = utils.normalize(
                    x[indexes], x_min=s_min, x_max=s_max
                )
    else:
        raise NotImplementedError(
            f"normalization mode {args.norm_mode} not implemented."
        )


def construct_dataset(args, x: np.ndarray, y: np.ndarray):
    """Construct feature-label pairs for the specified classification mode"""
    if args.classification_mode == 0:
        y = y[..., 0]  # extract session IDs
        # convert session IDs to class
        for session_ids, c in args.session2class.items():
            for session_id in session_ids:
                y = np.where(y == session_id, c, y)
    else:
        raise NotImplementedError(
            f"classification mode {args.classification_mode} not implemented."
        )
    return x, y


def get_datasets(args, buffer_size: int = 1024):
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"dataset {args.dataset} not found.")

    if args.verbose:
        print(f"\nloading data from {args.dataset}...")

    set_classification_mode(args)
    data, args.ds_info = preprocessing.preprocess(args)
    print(
        f'{args.config.split("/")[-1].split(".")[0]}: '
        f'percentage filtered in sessions upon QC -> '
        f'median = {np.median([v for v in args.ds_info["sessions_qc_percentage"].values()]):.03f}, '
        f'iqr = {stats.iqr([v for v in args.ds_info["sessions_qc_percentage"].values()]):.03f}\n'
    )
    normalize_features(args, data=data)

    x_train, y_train = construct_dataset(args, x=data["x_train"], y=data["y_train"])
    x_val, y_val = construct_dataset(args, x=data["x_val"], y=data["y_val"])
    x_test, y_test = construct_dataset(args, x=data["x_test"], y=data["y_test"])

    args.input_shape = x_train.shape[1:]
    assert args.input_shape[-1] == len(args.ds_info["channel_names"])

    args.train_steps = ceil(len(x_train) / args.batch_size)
    args.val_steps = ceil(len(x_val) / args.batch_size)
    args.test_steps = ceil(len(x_test) / args.batch_size)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(buffer_size)
    train_ds = train_ds.batch(args.batch_size)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(args.batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(args.batch_size)

    dry_run(datasets=[train_ds, val_ds, test_ds])

    return train_ds, val_ds, test_ds


def scramble_test_ds(args, x_test, y_test, to_permute: t.List = None):
    features = x_test.copy()
    if to_permute is not None:
        if len(to_permute) > 1:
            start, end = to_permute[0], to_permute[-1]
            np.random.seed(1234)
            np.random.shuffle(features[:, :, start:end])
        else:
            np.random.seed(1234)
            np.random.shuffle(features[:, :, to_permute[0]])
    test_ds = tf.data.Dataset.from_tensor_slices((features, y_test))
    test_ds = test_ds.batch(args.batch_size)

    dry_run(datasets=[test_ds])

    return test_ds
