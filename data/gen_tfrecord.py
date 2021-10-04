import json
import sys
import tensorflow as tf
import gflags
import logging
from io import BytesIO
from PIL import Image
from gen_annotation_json import get_annotation_dict

logger = logging.getLogger(__name__)

Flags = gflags.FLAGS
gflags.DEFINE_string('images_dir', './quick-draw-images', 'image path')
gflags.DEFINE_string('word2num_path', './word2num.json', 'word2num json path')
gflags.DEFINE_string('annotation_path', './annotation.json', 'annotation path')
gflags.DEFINE_string('record_path', './quick-draw-train.record', "record path")

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def process_image(image, resize):
    w, h = image.size
    if resize is not None:
        if w > h:
            w = int(w*resize/h)
            h = resize
        else:
            w = resize
            h = int(h*resize/w)
        image = image.resize((w,h), Image.ANTIALIAS)
    return image

def create_tf_example(image_path, label, resize=None):
    with open(image_path, "rb") as fid:
        image = Image.open(fid)
        image = process_image(image, resize)
        bytes_io = BytesIO()
        image.save(bytes_io, format='PNG')
        encoded_png = bytes_io.getvalue()
        w, h = image.size
        tf_example = tf.train.Example(
                features = tf.train.Features(
                    feature={
                        "image/encoded": bytes_feature(encoded_png),
                        "image/format": bytes_feature(b'png'),
                        "image/class/label": int64_feature(label),
                        "image/height": int64_feature(h),
                        "image/width": int64_feature(w)
                        }
                    ))
        return tf_example

def gen_tfrecord(annotation_dict, record_path, resize=None):
    num_tf_example = 0
    writer = tf.io.TFRecordWriter(record_path)
    for image_path, label in annotation_dict.items():
        if not tf.io.gfile.GFile(image_path):
            logger.error("{} does not exist".format(image_path))
        tf_example = create_tf_example(image_path, label, resize)
        writer.write(tf_example.SerializeToString())
        num_tf_example += 1
        if num_tf_example % 10000 == 0:
            logger.info("Create %d tf_example"%num_tf_example)
    writer.close()
    logger.info("{} tf_examples has been created successfully {}".format(num_tf_example, record_path))

def main(argv):
    Flags(argv)
    word2number_dict = None
    images_dir = Flags.images_dir
    record_path = Flags.record_path
    word2num_path = Flags.word2num_path
    with open(word2num_path, 'r') as json_file:
        word2number_dict = json.load(json_file)
    annotation_dict = get_annotation_dict(images_dir, word2number_dict)
    gen_tfrecord(annotation_dict, record_path)

if __name__ == '__main__':
    main(sys.argv)
