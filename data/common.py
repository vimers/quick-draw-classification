import tensorflow as tf
import json
import os
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

IMG_SIZE = 256
MODEL_INPUT_SHAPE = [IMG_SIZE, IMG_SIZE, 3]

MAX_IMG_NUM_PER_CATEGORY = 1000
FEATURE_DESC = (
        {
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "image/format": tf.io.FixedLenFeature([], tf.string),
            "image/class/label": tf.io.FixedLenFeature([], tf.int64),
            "image/height": tf.io.FixedLenFeature([], tf.int64),
            "image/width": tf.io.FixedLenFeature([], tf.int64)
            }
        )

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_feature(encoded_png, label, height=IMG_SIZE, width=IMG_SIZE):
    return {
            "image/encoded": bytes_feature(encoded_png),
            "image/format": bytes_feature(b"JPEG"),
            "image/class/label": int64_feature(label),
            "image/height": int64_feature(height),
            "image/width": int64_feature(width)
            }

def __decode_image(image, img_h, img_w):
    image = tf.image.decode_jpeg(image)
    image = tf.image.grayscale_to_rgb(image)
    if MODEL_INPUT_SHAPE[0] != img_h or MODEL_INPUT_SHAPE[1] != img_w:
        image = tf.image.resize(image, MODEL_INPUT_SHAPE[:2])
    else:
        image = tf.cast(image, tf.float32)
    return image

def parse_example(example):
    logger.debug("before decode image")
    example = tf.io.parse_example(example, FEATURE_DESC)
    image = __decode_image(example["image/encoded"],
            example["image/height"], example["image/width"])
    label = tf.cast(example["image/class/label"], tf.int32)
    logger.debug("after decode image")
    return image, label

def get_class_num():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    json_path = os.path.join(cur_dir, 'word2num.json')
    with open(json_path, 'r') as f:
        word2num = json.load(f)
        return len(word2num)

def get_label_name_by_id(label_id):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    json_path = os.path.join(cur_dir, 'word2num.json')
    with open(json_path, 'r') as f:
        word2num = json.load(f)
        for name, idx in word2num.items():
            if idx == label_id:
                return name
    return None

def get_label_id_by_name(label_name):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    json_path = os.path.join(cur_dir, 'word2num.json')
    with open(json_path, 'r') as f:
        word2num = json.load(f)
        return word2num[label_name]
