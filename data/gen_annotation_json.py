import os

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
def get_annotation_dict(input_folder, word2number_dict):
    label_dict = {}
    parent_list = os.listdir(input_folder)
    for parent in parent_list:
        parent_path = os.path.join(input_folder, parent)
        img_file_list = os.listdir(parent_path)
        for img_name in img_file_list:
            label_dict[os.path.join(parent_path, img_name)] = word2number_dict[parent]

    return label_dict
