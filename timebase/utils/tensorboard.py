import os
import io
import platform
import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt


def save_figure(figure: plt.Figure, filename: str, dpi: int = 120, close: bool = True):
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    figure.savefig(
        filename, dpi=dpi, bbox_inches="tight", pad_inches=0.01, transparent=True
    )
    if close:
        plt.close(figure)


class Summary(object):
    """Helper class to write TensorBoard summaries"""

    def __init__(self, args, output_dir: str = ""):
        self.dpi = args.dpi
        self.format = args.format
        self.dataset = args.dataset
        self.save_plots = args.save_plots
        self.class2name = args.class2name

        # write TensorBoard summary to specified output_dir or args.output_dir
        if output_dir:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            self.writers = [tf.summary.create_file_writer(output_dir)]
        else:
            output_dir = args.output_dir
            self.writers = [
                tf.summary.create_file_writer(output_dir),
                tf.summary.create_file_writer(os.path.join(output_dir, "val")),
                tf.summary.create_file_writer(os.path.join(output_dir, "test")),
            ]

        self.plots_dir = os.path.join(output_dir, "plots")
        if not os.path.isdir(self.plots_dir):
            os.makedirs(self.plots_dir)

        if platform.system() == "Darwin" and args.verbose == 2:
            matplotlib.use("TkAgg")

    def get_writer(self, mode: int = 0):
        return self.writers[mode]

    def close(self):
        for writer in self.writers:
            writer.close()

    def scalar(self, tag, value, step: int = 0, mode: int = 0):
        writer = self.get_writer(mode)
        with writer.as_default():
            tf.summary.scalar(tag, value, step=step)

    def histogram(self, tag, values, step: int = 0, mode: int = 0):
        writer = self.get_writer(mode)
        with writer.as_default():
            tf.summary.histogram(tag, values, step=step)

    def image(self, tag, values, step: int = 0, mode: int = 0):
        writer = self.get_writer(mode)
        with writer.as_default():
            tf.summary.image(tag, data=values, step=step, max_outputs=len(values))

    def figure(
        self,
        tag: str,
        figure: plt.Figure,
        step: int = 0,
        close: bool = True,
        mode: int = 0,
    ):
        """Write matplotlib figure to summary
        Args:
          tag: str, data identifier
          figure: plt.Figure, matplotlib figure or a list of figures
          step: int, global step value to record
          close: bool, close figure if True
          mode: int, indicate which summary writers to use
        """
        if self.save_plots:
            save_figure(
                figure,
                filename=os.path.join(
                    self.plots_dir, f"epoch_{step:03d}", f"{tag}.{self.format}"
                ),
                dpi=self.dpi,
                close=False,
            )
        buffer = io.BytesIO()
        figure.savefig(
            buffer, dpi=self.dpi, format="png", bbox_inches="tight", pad_inches=0.02
        )
        buffer.seek(0)
        image = tf.image.decode_png(buffer.getvalue(), channels=4)
        self.image(tag, tf.expand_dims(image, 0), step=step, mode=mode)
        if close:
            plt.close(figure)
