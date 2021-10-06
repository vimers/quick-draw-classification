import logging
import os
import sys

import gflags
import tensorflow as tf
from tensorflow import keras

from data.common import parse_example
from model import make_or_restore_model

logging.basicConfig()
logger = logging.getLogger(__name__)

EVALUATE_SIZE = 1280

Flags = gflags.FLAGS
gflags.DEFINE_string("tfrecord_path", "./quick-draw-train.record", "tfrecord file path")
gflags.DEFINE_string("ckpt_path", "./ckpt", "checkpoint dir path")
gflags.DEFINE_bool("train", True, "train or test")

TRAIN_BATCH = 16

def main(argv):
    try:
        Flags(argv)
    except gflags.FlagsError as e:
        logger.error("%s\nUsage: %s ARGS\n%s", e, argv[0], Flags)
        sys.exit(-1)
    ckpt_dir = Flags.ckpt_path
    if not os.path.exists(ckpt_dir):
        logger.info("ckpt dir %s not exist, create", ckpt_dir)
        os.makedirs(Flags.ckpt_path)
    full_dataset = tf.data.TFRecordDataset([Flags.tfrecord_path])
    full_dataset_size = sum(1 for _ in full_dataset)
    logger.debug("full_dataset size %d", full_dataset_size)
    full_dataset = full_dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    full_dataset = full_dataset.shuffle(buffer_size=1024).repeat().batch(TRAIN_BATCH)
    model = make_or_restore_model(Flags.ckpt_path)
    if model is None:
        logger.error("model is None")
        sys.exit(-1)
    if Flags.train:
        callbacks = [
                keras.callbacks.ModelCheckpoint(
                    filepath=ckpt_dir + "/ckpt-loss={loss:.2f}", save_freq=100
                    ),
                keras.callbacks.TensorBoard(
                    log_dir="./logs", histogram_freq=1, update_freq="batch"
                    )
                ]
        logger.info("========train========")
        train_dataset = full_dataset
        model.fit(train_dataset, batch_size=TRAIN_BATCH, epochs=100,
                steps_per_epoch=full_dataset_size/TRAIN_BATCH, callbacks=callbacks)
    else:
        logger.info("========evaluate========")
        evaluate_dataset = full_dataset.take(EVALUATE_SIZE)
        logger.debug("evaluate dataset num: %d", len(list(evaluate_dataset)))
        result = model.evaluate(evaluate_dataset, batch_size=TRAIN_BATCH, verbose=1)  # type: ignore
        logger.info("test loss %f test acc: %f", result[0], result[1])

if __name__ == '__main__':
    main(sys.argv)
