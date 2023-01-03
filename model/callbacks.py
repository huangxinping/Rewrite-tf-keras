import tensorflow as tf
import os
from pathlib import Path

class RewriteModelSaveCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, freq=2, directory='checkpoints', **kwargs):
        super().__init__()
        assert freq > 0
        self.freq = freq
        self.directory = directory
    
    def on_epoch_end(self, epoch, logs=None):
        os.makedirs(Path(os.getcwd(), self.directory), exist_ok=True)
        if epoch % self.freq == 0:
            self.model.save_weights(Path(self.directory, f"rewrite_{epoch}.h5"))
