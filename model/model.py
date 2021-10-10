import logging
import os

from tensorflow import keras
from tensorflow.keras import layers
from data.common import MODEL_INPUT_SHAPE, get_class_num

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def __get_uncompiled_model():
    inputs = keras.Input(MODEL_INPUT_SHAPE)
    base_model = keras.applications.MobileNet(input_shape=MODEL_INPUT_SHAPE, alpha=1.0,
            include_top=False, weights='imagenet')
    base_model.trainable = False
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    output = keras.layers.Dense(get_class_num(), activation='softmax')(x)
    model = keras.models.Model(inputs, output)
    model.summary()
    return model

def __get_compiled_model():
    model = __get_uncompiled_model()
    model.compile(optimizer="adam",
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
