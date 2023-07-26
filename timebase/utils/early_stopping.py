import os
import numpy as np
import tensorflow as tf

from timebase.utils import utils, yaml


class EarlyStopping:
    """Early Stopping module
    The monitor function monitors the validation loss and compare it against the
    previous best loss recorded. After min_epochs number of epochs, if the
    current loss value has not improved for more than patience number of epoch,
    then send terminate flag and load the best weight for the model.
    """

    def __init__(
        self,
        args,
        model: tf.keras.Model,
        checkpoint: tf.train.Checkpoint,
        patience: int = 10,
        min_epochs: int = 50,
    ):
        self.args = args
        self.model = model
        self.checkpoint = checkpoint
        self.patience = patience
        self.min_epochs = min_epochs

        self.wait = 0
        self.best_loss = np.inf
        self.best_epoch = -1
        self.best_weights = None

    def monitor(self, loss: float, epoch: int):
        terminate = False
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            self.wait = 0
            path = utils.save_checkpoint(
                self.args,
                checkpoint=self.checkpoint,
                epoch=epoch,
                filename="best_model",
            )
            yaml.save(
                filename=os.path.join(self.args.checkpoint_dir, "best_model.yaml"),
                data={"epoch": epoch, "val_loss": loss, "path": os.path.basename(path)},
            )
        elif epoch > self.min_epochs:
            if self.wait < self.patience:
                self.wait += 1
            else:
                terminate = True
                self.model.set_weights(self.best_weights)
                if self.args.verbose:
                    print(
                        f"EarlyStopping: model has not improved in {self.wait} epochs.\n"
                    )
        return terminate

    def restore(self):
        """Restore the best weights to the model"""
        self.model.set_weights(self.best_weights)
