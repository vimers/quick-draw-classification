import logging
import os

from tensorflow import keras
from tensorflow.keras import layers

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def __get_uncompiled_model():
    model = keras.Sequential()
    model.add(keras.Input(shape=(256, 256, 1)))
    model.add(layers.Conv2D(16, 5, strides=2))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(32, 5))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPool2D(3))
    model.add(layers.Conv2D(32, 5))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(32, 5))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(32, 5))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(32, 5))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(32, 5))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(32, 5))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(32, 5))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(64, 5))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(64, 3))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(128, 3))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.GlobalMaxPool2D())
    model.add(layers.Dense(68))
    model.add(layers.Softmax())
    # print model structure
    model.summary()
    return model

def __get_compiled_model():
    model = __get_uncompiled_model()
    model.compile(optimizer="rmsprop",
            loss="sparse_categorical_crossentropy",
            metrics="sparse_categorical_accuracy")
    return model

def make_or_restore_model(ckpt_dir):
    ckpts = [ckpt_dir + "/" + name for name in os.listdir(ckpt_dir)]
    if ckpts:
        latest_ckpt = max(ckpts, key=os.path.getctime)
        logger.info("Restore from %s", latest_ckpt)
        return keras.models.load_model(latest_ckpt)
    logger.info("Creating a new model")
    return __get_compiled_model()
