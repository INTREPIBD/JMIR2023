import typing as t
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.cm as cm
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics

from timebase.utils import tensorboard, utils

sns.set_style("ticks")
plt.style.use("seaborn-deep")

PARAMS_PAD = 2
PARAMS_LENGTH = 3
title_fontsize, label_fontsize, tick_fontsize = 14, 12, 10

plt.rcParams.update(
    {
        "mathtext.default": "regular",
        "xtick.major.pad": PARAMS_PAD,
        "ytick.major.pad": PARAMS_PAD,
        "xtick.major.size": PARAMS_LENGTH,
        "ytick.major.size": PARAMS_LENGTH,
    }
)

TICKER_FORMAT = matplotlib.ticker.FormatStrFormatter("%.2f")

JET = cm.get_cmap("jet")
GRAY = cm.get_cmap("gray")
TURBO = cm.get_cmap("turbo")
COLORMAP = TURBO
GRAY2RGB = COLORMAP(np.arange(256))[:, :3]


def remove_spines(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)


def remove_top_right_spines(axis):
    """Remove the ticks and spines of the top and right axis"""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def set_right_label(axis, label: str, fontsize: int = None):
    """Set y-axis label on the right-hand side"""
    right_axis = axis.twinx()
    kwargs = {"rotation": 270, "va": "center", "labelpad": 3}
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    right_axis.set_ylabel(label, **kwargs)
    right_axis.set_yticks([])
    remove_top_right_spines(right_axis)


def set_xticks(
    axis,
    ticks_loc: t.Union[np.ndarray, list],
    ticks: t.Union[np.ndarray, list],
    label: str = "",
    tick_fontsize: int = None,
    label_fontsize: int = None,
):
    axis.set_xticks(ticks_loc)
    axis.set_xticklabels(ticks, fontsize=tick_fontsize)
    if label:
        axis.set_xlabel(label, fontsize=label_fontsize)


def set_yticks(
    axis: matplotlib.axes.Axes,
    ticks_loc: t.Union[np.ndarray, list],
    ticks: t.Union[np.ndarray, list],
    label: str = "",
    tick_fontsize: int = None,
    label_fontsize: int = None,
):
    axis.set_yticks(ticks_loc)
    axis.set_yticklabels(ticks, fontsize=tick_fontsize)
    if label:
        axis.set_ylabel(label, fontsize=label_fontsize)


def plot_samples(
    args,
    ds: tf.data.Dataset,
    summary: tensorboard.Summary,
    epoch: int,
    mode: int,
    num_samples: int = 5,
):
    """Plot the first sample in num_samples batches in dataset ds"""
    if epoch == 0:
        channel_names = args.ds_info["channel_names"]
        for i, (x, y) in enumerate(ds.take(num_samples)):
            x, y = x.numpy()[0], y.numpy()[0]
            nrows = x.shape[1]
            figure, axes = plt.subplots(
                nrows=nrows,
                ncols=1,
                sharex=True,
                gridspec_kw={"wspace": 0.1, "hspace": 0.2},
                figsize=(4.5, 7.5),
                dpi=args.dpi,
            )
            for c in range(nrows):
                axes[c].plot(x[:, c], linewidth=1.5)
            for j, ax in enumerate(axes):
                remove_top_right_spines(axis=ax)
                set_right_label(axis=ax, label=channel_names[j])
            axes[0].set_title(args.class2name[y])
            axes[-1].set_xlabel("Time (s)")

            summary.figure(tag=f"samples/{i:03d}", figure=figure, step=epoch, mode=mode)


def confusion_matrix(
    args,
    tag: str,
    summary: tensorboard.Summary,
    y_true: t.Union[tf.Tensor, np.ndarray],
    y_pred: t.Union[tf.Tensor, np.ndarray],
    epoch: int,
    mode: int,
):

    labels = list(range(len(args.class2name)))
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
        figsize=(7, 7),
        dpi=args.dpi,
    )
    cbar_ax = figure.add_axes([0.92, 0.3, 0.02, 0.45])

    cmap = sns.color_palette("rocket_r", as_cmap=True)
    sns.heatmap(
        cm / np.sum(cm, axis=-1)[:, np.newaxis],
        vmin=0,
        vmax=1,
        cmap=cmap,
        annot=cm.astype(str),
        fmt="",
        linewidths=0.01,
        cbar=True,
        cbar_ax=cbar_ax,
        ax=ax,
    )

    ax.set_xlabel(f"Predictions", fontsize=label_fontsize)
    ax.set_ylabel("Targets", fontsize=label_fontsize)
    ax.set_title(
        f"Accuracy: {metrics.accuracy_score(y_true, y_pred):.04f}  | "
        f"F1-score: {metrics.f1_score(y_true, y_pred, labels=labels, average='macro'):.04f}",
        fontsize=title_fontsize,
    )

    ticklabels = [args.class2name[i] for i in range(len(args.class2name))]
    ax.set_xticklabels(
        ticklabels, va="top", ha="center", rotation=90, fontsize=tick_fontsize
    )
    ax.set_yticklabels(
        ticklabels, va="center", ha="right", rotation=0, fontsize=tick_fontsize
    )
    ax.tick_params(axis="both", which="both", length=0, pad=PARAMS_PAD + 2)

    cbar_ticks_loc = np.linspace(0, 1, 4)
    set_yticks(
        axis=cbar_ax,
        ticks_loc=cbar_ticks_loc,
        ticks=np.round(cbar_ticks_loc, 1),
        tick_fontsize=tick_fontsize,
    )
    cbar_ax.tick_params(axis="both", which="both", length=0, pad=PARAMS_PAD)

    summary.figure(tag, figure=figure, step=epoch, mode=mode)


def plot_auroc(
    args,
    tag: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    summary: tensorboard.Summary,
    epoch: int,
    mode: int,
):

    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    one_hot_y_true = np.zeros((y_true.size, int(y_true.max()) + 1), dtype=int)
    one_hot_y_true[np.arange(y_true.size), y_true.astype(int)] = 1

    # micro-average
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(
        one_hot_y_true.ravel(), y_score.ravel()
    )
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # macro-average
    n_classes = one_hot_y_true.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(one_hot_y_true[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # as test set is designed to be perfectly balanced, micro- and macro-average coincide

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
        figsize=(7, 7),
        dpi=args.dpi,
    )
    ax.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    ax.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    ax.axis("square")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic - One-vs-Rest Multiclass")
    plt.legend()

    summary.figure(tag=tag, figure=figure, step=epoch, mode=mode)


def model_res(
    args,
    ds,
    model,
    summary: tensorboard.Summary,
    epoch: int,
    mode: int = 1,
    results: t.Dict = None,
):
    y_true, y_pred, y_score = [], [], []
    for x, y in ds:
        y_true.append(y)
        outputs = model(x, training=False)
        y_pred.append(tf.argmax(outputs, axis=-1))
        y_score.append(outputs)
    y_true = tf.concat(y_true, axis=0).numpy().astype(int)
    y_pred = tf.concat(y_pred, axis=0).numpy().astype(int)
    y_score = tf.concat(y_score, axis=0).numpy()
    confusion_matrix(
        args,
        tag="confusion_matrix",
        summary=summary,
        y_true=y_true,
        y_pred=y_pred,
        epoch=epoch,
        mode=mode,
    )
    plot_auroc(
        args,
        tag="auroc",
        summary=summary,
        y_true=y_true,
        y_score=y_score,
        epoch=epoch,
        mode=mode,
    )
    plot_samples(
        args,
        ds=ds,
        summary=summary,
        epoch=epoch,
        mode=mode,
    )
    if results is not None:
        res = {
            "precision": metrics.precision_score(
                y_true, y_pred, average="macro", labels=np.unique(y_true)
            ),
            "recall": metrics.recall_score(
                y_true, y_pred, average="macro", labels=np.unique(y_true)
            ),
            "f1_score": metrics.f1_score(
                y_true, y_pred, average="macro", labels=np.unique(y_true)
            ),
            "roc_auc": metrics.roc_auc_score(
                y_true, y_score, average="macro", multi_class="ovr"
            ),
        }
        for k, v in res.items():
            results[k] = np.float32(v)
