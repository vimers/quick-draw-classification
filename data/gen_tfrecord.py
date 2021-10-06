from io import BytesIO
import json
import logging
import sys

from PIL import Image
import gflags
import progressbar
import tensorflow as tf
from data.common import create_feature

from gen_annotation_dict import gen_annotation_dict

logger = logging.getLogger(__name__)

Flags = gflags.FLAGS
gflags.DEFINE_string('images_dir', './quick-draw-images', 'image path')
gflags.DEFINE_string('record_path', './quick-draw-train.record', "record path")

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
                    feature=create_feature(encoded_png, label)
                    ))
        return tf_example

def gen_tfrecord(annotation_dict, record_path, resize=None):
    num_tf_example = 0
    writer = tf.io.TFRecordWriter(record_path)
    bar = progressbar.ProgressBar(
            maxval=len(annotation_dict),
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
            )
    bar.start()
    index = 0
    for image_path, label in annotation_dict.items():
        bar.update(value=index)
        index += 1
        if not tf.io.gfile.GFile(image_path):
            logger.error("{} does not exist".format(image_path))
        tf_example = create_tf_example(image_path, label, resize)
        writer.write(tf_example.SerializeToString())  # type: ignore
        num_tf_example += 1
        if num_tf_example % 10000 == 0:
            logger.info("Create %d tf_example"%num_tf_example)
    bar.finish()
    writer.close()
    logger.info("{} tf_examples has been created successfully {}".format(num_tf_example, record_path))

def main(argv):
    try:
        Flags(argv)
    except gflags.FlagsError as e:
        logger.error("%s\nUsage: %s ARGS\n%s", e, argv[0], Flags)
        sys.exit(-1)
    images_dir = Flags.images_dir
    record_path = Flags.record_path
    annotation_dict = gen_annotation_dict(images_dir)
    gen_tfrecord(annotation_dict, record_path)

if __name__ == '__main__':
    main(sys.argv)
