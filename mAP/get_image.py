import os
import shutil


image_infos = open("../config/test.txt").read().strip().split('\n')

if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

for image_info in image_infos:
    image_boxes = image_info.split(' ')
    image = image_boxes[0]

    target_path = os.path.join("./input/images-optional", os.path.basename(image))
    shutil.copy(image, target_path)
