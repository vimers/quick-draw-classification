import os
import logging
from data.common import get_label_id_by_name, MAX_IMG_NUM_PER_CATEGORY

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

"""
param:
input_folder: the dir path that contains multi category subfolder
word2number_dict: the map from category name to index

return dict just like this:
{
    "/xxx/xxx.png": 0,
    "/xxx/xxx.png": 2
}
"""
def gen_annotation_dict(input_folder):
    label_dict = {}
    parent_list = os.listdir(input_folder)
    for parent in parent_list:
        parent_path = os.path.join(input_folder, parent)
        img_file_list = os.listdir(parent_path)
        logger.debug("%s: %d", parent, get_label_id_by_name(parent))
        for index, img_name in enumerate(img_file_list):
            if index > MAX_IMG_NUM_PER_CATEGORY:
                break
            label_dict[os.path.join(parent_path, img_name)] = get_label_id_by_name(parent)
    return label_dict
