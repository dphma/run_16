import numpy as np
import config.config as cfg


def generate_anchors(sizes=None, ratios=None):
   
    if ratios is None:
        ratios = cfg.anchor_box_ratios
    if sizes is None:
        sizes = cfg.anchor_box_scales

    num_anchors = len(sizes) * len(ratios)

    anchors = np.zeros((num_anchors, 4), dtype=np.float32)

    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T
  
    # [[  0.   0. 128. 128.]
    #  [  0.   0. 256. 256.]
    #  [  0.   0. 512. 512.] 

    for i in range(len(ratios)):
        # anchor 1:1，1:2，2:1
        anchors[3 * i:3 * i + 3, 2] = anchors[3 * i:3 * i + 3, 2] * ratios[i][0]
        anchors[3 * i:3 * i + 3, 3] = anchors[3 * i:3 * i + 3, 3] * ratios[i][1]

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(share_layer_shape, anchors, stride=cfg.rpn_stride):
  
    coordinate_x = (np.arange(0, share_layer_shape[0], dtype=np.float32) + 0.5) * stride
    coordinate_y = (np.arange(0, share_layer_shape[1], dtype=np.float32) + 0.5) * stride

    coordinate_x, coordinate_y = np.meshgrid(coordinate_x, coordinate_y)

    coordinate_x = np.reshape(coordinate_x, [-1])
    coordinate_y = np.reshape(coordinate_y, [-1])

    coordinates = np.stack([
        coordinate_x,
        coordinate_y,
        coordinate_x,
        coordinate_y
    ], axis=0)

    coordinates = np.transpose(coordinates)

    number_of_anchors = np.shape(anchors)[0]
    k = np.shape(coordinates)[0]

    shifted_anchors = np.expand_dims(anchors, axis=0) + np.expand_dims(coordinates, axis=1)
   
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors


def get_anchors(share_layer_shape, image_shape):
    width, height = image_shape
    anchors = generate_anchors()
    network_anchors = shift(share_layer_shape, anchors)

    network_anchors[:, 0] = network_anchors[:, 0] / width
    network_anchors[:, 1] = network_anchors[:, 1] / height
    network_anchors[:, 2] = network_anchors[:, 2] / width
    network_anchors[:, 3] = network_anchors[:, 3] / height

    network_anchors = np.clip(network_anchors, 0, 1)

    return network_anchors

