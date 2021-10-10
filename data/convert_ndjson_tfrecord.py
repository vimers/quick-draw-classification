import json
import logging
import sys

import gflags
import ndjson
import progressbar
import tensorflow as tf
import cv2
import numpy as np

from common import create_feature


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

Flags = gflags.FLAGS
gflags.DEFINE_string("ndjson_path", "./quick-draw-ndjson", "Quick draw ndjson path")
gflags.DEFINE_string('record_path', './quick-draw-train.record', "record path")
gflags.DEFINE_string("word2num_path", "./data/word2num.json", "word2num path")

IMG_SIZE = 256
LINE_WIDTH = 6

def draw_stroke(item):
    img = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
    raw_strokes = item['drawing']
    for stroke in raw_strokes:
        for i in range(len(stroke[0])-1):
            cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i+1], stroke[1][i+1]), 255, LINE_WIDTH)
    img = img/255.
    img = img[:,:,np.newaxis]
    return cv2.imencode(".jpg", img)[1].tostring()

def create_tf_example(image, label):
    tf_example = tf.train.Example(
            features = tf.train.Features(
                feature=create_feature(image, label)
                ))
    return tf_example

MAX_IMG_PER_CATEGORY = 1000

def convert_ndjson_by_category(record_writer, label, label_id):
    with open('{}/{}.ndjson'.format(Flags.ndjson_path, label), 'r') as f:
        items = ndjson.load(f)
        num_recognized_item = sum(1 for item in items if item['recognized'])
        num_recognized_item = max(num_recognized_item, MAX_IMG_PER_CATEGORY)
        bar = progressbar.ProgressBar(maxval=num_recognized_item,
                widgets=[progressbar.Bar('=', '{}['.format(label), ']'), ' ', progressbar.Percentage()])
        bar.start()
        for index, item in enumerate(items):
            assert item['word'] == label
            if not item['recognized']:
                logger.debug('{} not recognized'.format(item['key_id']))
                continue
            img = draw_stroke(item)
            example = create_tf_example(img, label_id)
            record_writer.write(example.SerializeToString())  # type: ignore
            bar.update(index)
            if index > MAX_IMG_PER_CATEGORY:
                break
        bar.finish()

def main(argv):
    try:
        Flags(argv)
    except gflags.FlagsError as e:
        logger.error("%s\nusage: %s args\n%s", e, argv[0], Flags)
        sys.exit(-1)
    word2num = None
    record_path = Flags.record_path
    record_writer = tf.io.TFRecordWriter(record_path)
    with open(Flags.word2num_path, "r") as json_file:
        word2num = json.loads(json_file.read())
        for label, label_id in word2num.items():
            convert_ndjson_by_category(record_writer, label, label_id)
    record_writer.close()

if __name__ == '__main__':
    main(sys.argv)
