import os
import sys
import config.config as cfg

image_infos = open("../config/test.txt").read().strip().split('\n')


if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/ground-truth"):
    os.makedirs("./input/ground-truth")

for image_info in image_infos:
    image_boxes = image_info.split(' ')
    image, boxes = image_boxes[0], image_boxes[1:]

    with open("./input/ground-truth/" + os.path.basename(image)[:-4] + ".txt", "w") as file:
        for box in boxes:
            xmin, ymin, xmax, ymax, idx = box.split(',')
            name = cfg.label[int(idx)]
            file.write("{} {} {} {} {}\n".format(name, xmin, ymin, xmax, ymax))

print("Conversion completed!")
