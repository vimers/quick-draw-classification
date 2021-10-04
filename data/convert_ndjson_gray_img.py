#  import numpy as np
import os
import sys
import json
import gflags
import progressbar
import ndjson
import logging
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

Flags = gflags.FLAGS
gflags.DEFINE_string("image_path", "./quick-draw-images", "Base image path")
gflags.DEFINE_string("ndjson_path", "./quick-draw-ndjson", "Quick draw ndjson path")
gflags.DEFINE_string("word2num_path", "./word2num.json", "word2num path")
gflags.DEFINE_bool("gen_unknown", True, "whether to gen unknown")

def draw_stroke(category_dir, item):
    img = Image.new("L", color="white", size=(256,256))
    draw = ImageDraw.Draw(img)
    gray_file = os.path.join(category_dir, '{}.png'.format(item['key_id']))
    for stroke in item['drawing']:
        draw.line(list(zip(stroke[0], stroke[1])), fill="black", width=3)
    with open(gray_file, "wb") as f:
        img.save(f, "PNG")

def convert_ndjson_by_category(category):
    with open('{}/{}.ndjson'.format(Flags.ndjson_path, category), 'r') as f:
        items = ndjson.load(f)
        category_dir = os.path.join(Flags.image_path, category)
        os.makedirs(category_dir, exist_ok=True)
        num_recognized_item = sum(1 for item in items if item['recognized'])
        bar = progressbar.ProgressBar(maxval=num_recognized_item,
                widgets=[progressbar.Bar('=', '{}['.format(category), ']'), ' ', progressbar.Percentage()])
        bar.start()
        record_index = 0
        for item in items:
            assert item['word'] == category
            if not item['recognized']:
                logger.debug('{} not recognized'.format(item['key_id']))
                continue
            draw_stroke(category_dir, item)
            record_index = record_index + 1
            bar.update(record_index)
        bar.finish()

NUM_UNKNOWN_IMGS = 120000
NAME_UNKNOWN_CATEGORY = "unknown"
def convert_unknown_ndjson():
    ndjson_files = os.listdir(Flags.ndjson_path)
    num_ndjson = sum(1 for f in ndjson_files if f.endswith(".ndjson"))
    num_unknown_each_category = NUM_UNKNOWN_IMGS/num_ndjson;
    for ndjson_file in ndjson_files:
        if not ndjson_file.endswith(".ndjson"):
            logger.warning("ndjson dir contains {}, not process".format(ndjson_file))
            break
        with open("{}/{}".format(Flags.ndjson_path, ndjson_file), "r") as f:
            items = ndjson.load(f)
            unknown_dir = os.path.join(Flags.image_path, NAME_UNKNOWN_CATEGORY)
            os.makedirs(unknown_dir, exist_ok=True)
            unknown_items = 0
            for item in items:
                if item['recognized']:
                    logger.debug('{} recognized'.format(item['key_id']))
                    continue
                unknown_items += 1
                draw_stroke(unknown_dir, item)
                if unknown_items > num_unknown_each_category:
                    logger.debug("{} exceed max num".format(unknown_items))
                    break

def main(argv):
    Flags(argv)
    if Flags.gen_unknown:
        convert_unknown_ndjson()
    else:
        word2num = None
        with open(Flags.word2num_path, "r") as json_file:
            word2num = json.loads(json_file.read())
            for category in word2num.keys():
                img_dir = os.path.join(Flags.image_path, category)
                if os.path.isdir(img_dir):
                    logger.warn('{} already exist, ignore convert'.format(category))
                    continue
                else:
                    logger.info('begin convert {}'.format(category))
                    convert_ndjson_by_category(category)

if __name__ == '__main__':
    main(sys.argv)
