import tensorflow as tf
import json
import os

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


def create_feature(encoded_png, label, height=256, width=256):
    return {
            "image/encoded": bytes_feature(encoded_png),
            "image/format": bytes_feature(b"PNG"),
            "image/class/label": int64_feature(label),
            "image/height": int64_feature(height),
            "image/width": int64_feature(width)
            }

def __decode_image(image):
    image = tf.image.decode_png(image)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [256, 256, 1]) # NHWC
    return image

def parse_example(example):
    example = tf.io.parse_example(example, FEATURE_DESC)
    image = __decode_image(example["image/encoded"])
    label = tf.cast(example["image/class/label"], tf.int32)
    return image, label


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
