import os
import argparse
from math import ceil
from multiprocessing import Process

import classifier_train


class TrialArgs:
    def __init__(self, args, time_alignment: int, segment_length: int):
        self.output_dir = os.path.join(
            args.output_dir,
            f"ta{time_alignment:02d}_sl{segment_length:04d}",
        )

        self.batch_size = int(
            args.batch_size * ceil(8192 / (time_alignment * segment_length))
        )
        self.epochs = 100
        self.seed = args.seed
        self.mixed_precision = True

        self.dataset = args.dataset
        self.classification_mode = 0
        self.config = args.config
        self.downsampling = "average"
        self.time_alignment = time_alignment
        self.norm_mode = 1
        self.padding_mode = "average"
        self.filter_mode = 2
        self.ibi_interpolation = "quadratic"
        self.hrv_features = []
        self.hrv_length = 60
        self.segment_length = segment_length
        self.test_size = 20

        self.model = args.model
        self.num_units = 128
        self.activation = "gelu"
        self.lr = 0.001
        self.l2 = 0.0
        self.dropout = 0.0

        self.r_dropout = 0.0

        self.num_encoders = 4
        self.head_size = 256
        self.num_heads = 4
        self.ff_dim = 4
        self.t_dropout = 0.25

        self.save_plots = True
        self.format = "pdf"
        self.dpi = 120

        self.verbose = 0
        self.clear_output_dir = True


def main(args):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # define search space
    time_alignments = [1, 2, 4, 8, 16, 32]
    segment_lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    for time_alignment in time_alignments:
        for segment_length in segment_lengths:
            trial_args = TrialArgs(
                args,
                time_alignment=time_alignment,
                segment_length=segment_length,
            )
            if os.path.exists(os.path.join(trial_args.output_dir, "results.yaml")):
                print(f"Skipping trial : {trial_args.output_dir}")
                continue
            print(f"\nTrial {trial_args.output_dir}")
            process = Process(target=classifier_train.main, args=(trial_args,))
            process.start()
            process.join()

    print(f"Experiments saved at {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training configuration
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to .yaml file that contains classification labels",
    )

    # model configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/raw_data",
        help="path to directory with raw data in zip files",
    )
    parser.add_argument("--model", type=str, required=True)

    # matplotlib
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--format", type=str, default="pdf", choices=["pdf", "png", "svg"]
    )
    parser.add_argument("--save_plots", action="store_true")

    parser.add_argument("--seed", type=int, default=1234)

    main(parser.parse_args())