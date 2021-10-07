import logging
import os
import sys
import time

from dotenv import load_dotenv
import gflags
import notifiers
import tensorflow as tf
from tensorflow import keras

from data.common import parse_example
from model import make_or_restore_model

logging.basicConfig()
logger = logging.getLogger(__name__)

NUM_EPOCHS = 100
EVALUATE_SIZE = 1280

Flags = gflags.FLAGS
gflags.DEFINE_string("tfrecord_path", "./quick-draw-train.record", "tfrecord file path")
gflags.DEFINE_string("ckpt_path", "./ckpt", "checkpoint dir path")
gflags.DEFINE_bool("train", True, "train or test")

TRAIN_BATCH = 16

class GitterNotify(keras.callbacks.Callback):
    def __init__(self):
        self.gitter = notifiers.get_notifier('gitter')
        dotenv = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".env")
        load_dotenv(dotenv)
        self.token = os.getenv("GITTER_TOKEN")
        self.room_id = os.getenv("ROOM_ID")
        if self.token is None or self.room_id is None:
            logger.error("gitter token and room id is None")
        self.message = "Epoch: {}/{}\nLoss: {}\nAccuracy: {}"
    def on_epoch_end(self, epoch, logs=None):
        msg = self.message.format(epoch+1, NUM_EPOCHS, round(logs['loss'], 3), round(logs['sparse_categorical_accuracy']*100, 3))
        print(self.token, self.room_id, msg)
        self.gitter.notify(token=self.token, room_id=self.room_id, message=msg)

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
                GitterNotify(),
                keras.callbacks.ModelCheckpoint(
                    filepath=ckpt_dir + "/ckpt.{epoch:02d}-{loss:.4f}-{sparse_categorical_accuracy:.4f}",
                    save_freq=1000
                    ),
                keras.callbacks.TensorBoard(
                    log_dir="./logs", histogram_freq=1, update_freq="batch"
                    )
                ]
        logger.info("========train========")
        train_dataset = full_dataset
        model.fit(train_dataset, batch_size=TRAIN_BATCH, epochs=NUM_EPOCHS,
                steps_per_epoch=full_dataset_size/TRAIN_BATCH, callbacks=callbacks)
    else:
        logger.info("========evaluate========")
        evaluate_dataset = full_dataset.take(EVALUATE_SIZE)
        logger.debug("evaluate dataset num: %d", len(list(evaluate_dataset)))
        result = model.evaluate(evaluate_dataset, batch_size=TRAIN_BATCH, verbose=1)  # type: ignore
        logger.info("test loss %f test acc: %f", result[0], result[1])

if __name__ == '__main__':
    main(sys.argv)
