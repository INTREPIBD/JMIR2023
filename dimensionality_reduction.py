import os
import shutil
import argparse
import numpy as np
import typing as t
from tqdm import tqdm
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import decomposition

from timebase.data import preprocessing
from timebase.utils import tensorboard, utils, yaml
from timebase.data.utils import unzip_session, shuffle


UNI_DEP_1 = "configs/config11.yaml"
UNI_DEP_2 = "configs/config12.yaml"

MAN_EPS_1 = "configs/config13.yaml"
MAN_EPS_2 = "configs/config14.yaml"

BI_DEP_1 = "configs/config15.yaml"
BI_DEP_2 = "configs/config16.yaml"

MIXED_MAN_1 = "configs/config17.yaml"
MIXED_MAN_2 = "configs/config18.yaml"

COLORS = sns.color_palette("Set2", 9)


def create_pairs(
    args,
    features: t.List[np.ndarray],
    label: t.List[int],
    num_samples: int = 20,
):
    """Segment features and return train, validation and test set pairs
    Returns:
      features: np.ndarray, segmented features
      labels: np.ndarray, paired labels
    """
    features = preprocessing.segmentation(args, features=features)
    features = np.random.permutation(features)[:num_samples]
    get_labels = lambda x: np.tile(label, reps=(len(x), 1)).astype(np.float32)
    return {"x": features, "y": get_labels(features)}


def get_data(args, config_filename: str):
    session2name = {
        tuple(v["id"]) if type(v["id"]) == list else (v["id"],): v["name"]
        for k, v in yaml.load(config_filename).items()
    }

    clinical_info = preprocessing.read_clinical_info(
        os.path.join(preprocessing.FILE_DIRECTORY, "TIMEBASE_database.xlsx")
    )
    data, sessions_info, channel_names = {}, {}, None
    for session_ids in tqdm(session2name.keys()):
        features, label = [], None
        for session_id in session_ids:
            recording_dir = unzip_session(args.dataset, session_id=session_id)
            s_features, s_label, session_info = preprocessing.preprocess_dir(
                args, recording_dir=recording_dir, clinical_info=clinical_info
            )
            sessions_info[session_id] = session_info
            features.extend(s_features)
            if label is None:
                label = s_label
            if channel_names is None:
                channel_names = session_info["channel_names"]
        session_data = create_pairs(args, features=features, label=label)
        utils.update_dict(data, session_data)

    data = {k: np.concatenate(v) for k, v in data.items()}
    data["x"], data["y"] = shuffle(data["x"], data["y"])

    ds_min = np.min(data["x"], axis=(0, 1))
    ds_max = np.max(data["x"], axis=(0, 1))
    ds_mean = np.mean(data["x"], axis=(0, 1))
    ds_std = np.std(data["x"], axis=(0, 1))
    # data["x"] = (data["x"] - ds_min) / ((ds_max - ds_min) + 1e-6)
    data["x"] = (data["x"] - ds_mean) / ds_std

    # convert data to shape (channels, num. samples, time-steps)
    data["x"] = np.transpose(data["x"], axes=(2, 0, 1))

    # data["x"] = np.stack(
    #     (
    #         np.min(data["x"], axis=-1),
    #         np.max(data["x"], axis=-1),
    #         np.mean(data["x"], axis=-1),
    #         np.var(data["x"], axis=-1),
    #         np.std(data["x"], axis=-1),
    #     ),
    #     axis=-1,
    # )
    data["y"] = data["y"][:, 0].astype(int)

    data_info = {"channel_names": channel_names, "session2name": session2name}
    return data, data_info


def get_label_name(session2name: t.Dict[tuple, str], session: int):
    for session_ids, label_name in session2name.items():
        for session_id in session_ids:
            if session == session_id:
                return label_name


def fit_pca(
    args,
    data: t.Dict[str, np.ndarray],
    filename: str,
    data_info: t.Dict,
    n_components: int = 2,
):
    n_channels = data["x"].shape[0]
    label_fontsize, tick_fontsize = 11, 9

    figure, axes = plt.subplots(
        nrows=1,
        ncols=n_channels,
        gridspec_kw={"wspace": 0.3, "hspace": 0.01},
        subplot_kw={"projection": None},
        figsize=(4.2 * n_channels, 3.5),
        dpi=args.dpi,
    )

    labels = sorted(np.unique(data["y"]).tolist())

    for c in range(n_channels):
        pca = decomposition.PCA(n_components=n_components)
        x_pc = pca.fit_transform(data["x"][c, ...])
        print(f"Channel {c} explained variance: {pca.explained_variance_ratio_}")
        for i, label in enumerate(labels):
            indexes = np.where(data["y"] == label)[0]
            axes[c].scatter(
                x_pc[indexes, 0],
                x_pc[indexes, 1],
                s=20,
                marker="x",
                alpha=0.9,
                color=COLORS[i],
                label=get_label_name(data_info["session2name"], label),
            )
        axes[c].set_title(data_info["channel_names"][c], fontsize=label_fontsize)
        axes[c].set_xlabel(
            rf"$PC_1$ (EV: {pca.explained_variance_ratio_[0]*100:.2f}%)",
            fontsize=label_fontsize,
        )
        axes[c].set_ylabel(
            rf"$PC_2$  (EV: {pca.explained_variance_ratio_[1]*100:.2f}%)",
            fontsize=label_fontsize,
        )
        axes[c].tick_params(axis="both", which="both", labelsize=tick_fontsize)
        tensorboard.remove_top_right_spines(axis=axes[c])

    axes[0].legend(
        loc="best",
        fontsize=tick_fontsize,
        handlelength=0.5,
        handletextpad=0.5,
        markerscale=0.8,
    )
    tensorboard.save_figure(figure, filename=filename, dpi=args.dpi)
    print(f"PCA plot saved to {filename}")


def main(args):
    tf.keras.utils.set_random_seed(args.seed)

    if args.clear_output_dir and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    data, data_info = get_data(args, config_filename=UNI_DEP_1)
    fit_pca(
        args,
        data=data,
        filename=os.path.join(args.output_dir, "uni_dep_1.pdf"),
        data_info=data_info,
    )

    data, data_info = get_data(args, config_filename=UNI_DEP_2)
    fit_pca(
        args,
        data=data,
        filename=os.path.join(args.output_dir, "uni_dep_2.pdf"),
        data_info=data_info,
    )

    data, data_info = get_data(args, config_filename=MAN_EPS_1)
    fit_pca(
        args,
        data=data,
        filename=os.path.join(args.output_dir, "man_eps_1.pdf"),
        data_info=data_info,
    )

    data, data_info = get_data(args, config_filename=MAN_EPS_2)
    fit_pca(
        args,
        data=data,
        filename=os.path.join(args.output_dir, "man_eps_2.pdf"),
        data_info=data_info,
    )

    data, data_info = get_data(args, config_filename=BI_DEP_1)
    fit_pca(
        args,
        data=data,
        filename=os.path.join(args.output_dir, "bi_dep_1.pdf"),
        data_info=data_info,
    )

    data, data_info = get_data(args, config_filename=BI_DEP_1)
    fit_pca(
        args,
        data=data,
        filename=os.path.join(args.output_dir, "bi_dep_2.pdf"),
        data_info=data_info,
    )

    data, data_info = get_data(args, config_filename=MIXED_MAN_1)
    fit_pca(
        args,
        data=data,
        filename=os.path.join(args.output_dir, "mixed_manic_1.pdf"),
        data_info=data_info,
    )

    data, data_info = get_data(args, config_filename=MIXED_MAN_2)
    fit_pca(
        args,
        data=data,
        filename=os.path.join(args.output_dir, "mixed_manic_2.pdf"),
        data_info=data_info,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # configuration
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("-seed", type=int, default=1234)

    # dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/raw_data",
        help="path to directory with raw data in zip files",
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
        choices=[1, 2, 4, 32, 64],
        help="number of samples per second (Hz) for time-alignment",
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
        default=1024,
        help="segmentation window length in seconds",
    )
    parser.add_argument("--test_segments", type=int, default=20)

    # matplotlib
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--format", type=str, default="pdf", choices=["pdf", "png", "svg"]
    )
    parser.add_argument("--save_plots", action="store_true")

    # misc
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--clear_output_dir", action="store_true")

    params = parser.parse_args()
    main(params)
