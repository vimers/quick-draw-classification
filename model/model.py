import os
import sys
import logging

import gflags
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Flags = gflags.FLAGS
gflags.DEFINE_string("tfrecord_path", "../data/quick-draw.record", "tfrecord file path")
gflags.DEFINE_string("ckpt_path", "./ckpt", "checkpoint dir path")

def get_uncompiled_model():
    model = keras.Sequential()
    model.add(keras.Input(shape=(256, 256, 1)))
    model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
    model.add(layers.Conv2D(32, 5, activation="relu"))
    model.add(layers.MaxPool2D(3))

    model.add(layers.Conv2D(32, 5, activation="relu"))
    model.add(layers.Conv2D(32, 5, activation="relu"))
    model.add(layers.Conv2D(32, 5, activation="relu"))
    model.add(layers.Conv2D(32, 5, activation="relu"))
    model.add(layers.Conv2D(32, 5, activation="relu"))
    model.add(layers.Conv2D(32, 5, activation="relu"))
    model.add(layers.Conv2D(32, 5, activation="relu"))
    model.add(layers.Conv2D(64, 5, activation="relu"))
    model.add(layers.Conv2D(64, 3, activation="relu"))
    model.add(layers.Conv2D(128, 3, activation="relu"))
    model.add(layers.GlobalMaxPool2D())
    model.add(layers.Dense(67))
    model.add(layers.Softmax())
    # print model structure
    model.summary()
    return model

def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(optimizer="rmsprop",
            loss="sparse_categorical_crossentropy",
            metrics="sparse_categorical_accuracy")
    return model

def decode_image(image):
    image = tf.image.decode_png(image)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [256, 256, 1])
    return image

def read_tfrecord(example):
    tfrecord_format = (
            {
                "image/encoded": tf.io.FixedLenFeature([], tf.string),
                "image/format": tf.io.FixedLenFeature([], tf.string),
                "image/class/label": tf.io.FixedLenFeature([], tf.int64),
                "image/height": tf.io.FixedLenFeature([], tf.int64),
                "image/width": tf.io.FixedLenFeature([], tf.int64)
                }
            )
    example = tf.io.parse_example(example, tfrecord_format)
    image = decode_image(example["image/encoded"])
    label = tf.cast(example["image/class/label"], tf.int32)
    return image, label

def make_or_restore_model():
    ckpt_dir = Flags.ckpt_path
    ckpts = [ckpt_dir + "/" + name for name in os.listdir(ckpt_dir)]
    if ckpts:
        latest_ckpt = max(ckpts, key=os.path.getctime)
        logger.info("Restore from %s", latest_ckpt)
        return keras.models.load_model(latest_ckpt)
    logger.info("Creating a new model")
    return get_compiled_model()

TRAIN_SIZE = 1024000
def main(argv):
    Flags(argv)
    ckpt_dir = Flags.ckpt_path
    if not os.path.exists(ckpt_dir):
        logger.info("ckpt dir %s not exist, create", ckpt_dir)
        os.makedirs(Flags.ckpt_path)
    full_dataset = tf.data.TFRecordDataset([Flags.tfrecord_path])
    full_dataset = full_dataset.map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    full_dataset = full_dataset.shuffle(buffer_size=1024).batch(64)
    train_dataset = full_dataset.take(TRAIN_SIZE)
    model = make_or_restore_model()
    if model is None:
        logger.error("model is None")
        sys.exit(-1)
    callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=ckpt_dir + "/ckpt-loss={loss:.2f}", save_freq=1000
                ),
            keras.callbacks.TensorBoard(
                log_dir="./logs", histogram_freq=1, update_freq="batch"
                )
            ]
    model.fit(train_dataset, batch_size=64, steps_per_epoch=TRAIN_SIZE/64, callbacks=callbacks)  # type: ignore

if __name__ == '__main__':
    main(sys.argv)
