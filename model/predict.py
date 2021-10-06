import logging

import sys
import cv2
import gflags
import numpy as np
from tensorflow import keras

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Flags = gflags.FLAGS
gflags.DEFINE_string("model_dir", "./ckpt", "model dir")
gflags.DEFINE_string("input", "./test.png", "predict image file")

def main(argv):
    Flags(argv)
    img = cv2.imread(Flags.input, cv2.IMREAD_GRAYSCALE)
    img = img.reshape(1, 256, 256, 1)
    model = keras.models.load_model(Flags.model_dir)
    result = model.predict(img)
    cls_index = np.argmax(result)
    logger.info("cls: %d conf: %f", cls_index, result[0, cls_index])

if __name__ == '__main__':
    main(sys.argv)
