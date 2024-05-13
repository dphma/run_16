from predict import FasterRCNN
from PIL import Image
import config.config as cfg
import os

import tensorflow as tf
from core.anchorGenerate import get_anchors
from core.boxParse import BoundingBox
import numpy as np
from tqdm import tqdm


class Frcnn(FasterRCNN):
    def __init__(self, weight_path):
        self.weight_path = weight_path
        super().__init__(weight_path)

    def detect_single_image(self, image, image_id):
        f = open("./input/detection-results/" + image_id + ".txt", "w")

        resize_image = self.process_image(image)
        old_w, old_h = image.size
        
        new_h, new_w = np.shape(resize_image)[1:-1]
        
        rpn_height, rpn_width = self.model_rpn.compute_output_shape((1, new_h, new_w, 3))[-1][1:-1]

        predict_rpn = self.model_rpn(resize_image, training=False)

        share_layer = predict_rpn[2]

        anchors = get_anchors(share_layer_shape=(rpn_width, rpn_height), image_shape=(new_w, new_h))
        box_parse = BoundingBox(anchors)
        predict_boxes = box_parse.detection_out(predict_rpn, anchors,  confidence_threshold=0)
        predict_boxes = predict_boxes[0]

        predict_boxes[:, 0] = np.array(np.round(predict_boxes[:, 0]*new_w/cfg.rpn_stride), dtype=np.int32)
        predict_boxes[:, 1] = np.array(np.round(predict_boxes[:, 1]*new_h/cfg.rpn_stride), dtype=np.int32)
        predict_boxes[:, 2] = np.array(np.round(predict_boxes[:, 2]*new_w/cfg.rpn_stride), dtype=np.int32)
        predict_boxes[:, 3] = np.array(np.round(predict_boxes[:, 3]*new_h/cfg.rpn_stride), dtype=np.int32)

        predict_boxes[:, 2] -= predict_boxes[:, 0]
        predict_boxes[:, 3] -= predict_boxes[:, 1]

        delete_line = []
        for i, r in enumerate(predict_boxes):
            if r[2] < 1 or r[3] < 1:
                delete_line.append(i)

        predict_boxes = np.delete(predict_boxes, delete_line, axis=0)

        boxes = []
        probs = []
        labels = []

        for i in range(predict_boxes.shape[0] // cfg.num_rois + 1):
            
            rois = predict_boxes[cfg.num_rois * i: cfg.num_rois * (i+1), :]
            rois = np.expand_dims(rois, axis=0)

            if rois.shape[1] == 0:
                print("Your model can't detect any bounding boxes.")
                exit(0)

            if i == predict_boxes.shape[0] // cfg.num_rois:

                curr_shape = rois.shape

                target_shape = (curr_shape[0], cfg.num_rois, curr_shape[2])

                rois_padded = np.zeros(target_shape).astype(rois.dtype)
                
                rois_padded[:, :curr_shape[1], :] = rois
                
                rois_padded[0, curr_shape[1]:, :] = rois[0, 0, :]

                rois = rois_padded

            p_cls, p_regr = self.model_classifier([share_layer, rois], training=False)

            for j in range(p_cls.shape[1]):
                
                conf = np.max(p_cls[0, j, :-1])
                if conf < self.confidence:
                    continue

                x, y, w, h = rois[0, j, :]

                cls_num = np.argmax(p_cls[0, j, :-1])

                tx, ty, tw, th = p_regr[0, j, 4 * cls_num: 4 * (cls_num + 1)]

                tx /= cfg.classifier_regr_std[0]
                ty /= cfg.classifier_regr_std[1]
                tw /= cfg.classifier_regr_std[2]
                th /= cfg.classifier_regr_std[3]

                cx = x + w / 2.0
                cy = y + h / 2.0
                
                cx1 = tx * w + cx
                cy1 = ty * h + cy
                w1 = np.exp(tw) * w
                h1 = np.exp(th) * h

                x1 = cx1 - w1 / 2.0
                y1 = cy1 - h1 / 2.0
                x2 = cx1 + w1 / 2.0
                y2 = cy1 + h1 / 2.0

                x1 = int(tf.round(x1))
                y1 = int(tf.round(y1))
                x2 = int(tf.round(x2))
                y2 = int(tf.round(y2))

                boxes.append([x1, y1, x2, y2])
                probs.append(conf)
                labels.append(cls_num)

        if len(boxes) == 0:
            return image

        labels = np.array(labels)
        probs = np.array(probs)
        boxes = np.array(boxes, dtype=np.float32)

        boxes[:, 0] = boxes[:, 0] * cfg.rpn_stride / new_w
        boxes[:, 1] = boxes[:, 1] * cfg.rpn_stride / new_h
        boxes[:, 2] = boxes[:, 2] * cfg.rpn_stride / new_w
        boxes[:, 3] = boxes[:, 3] * cfg.rpn_stride / new_h

        result = None
        for c in range(cfg.num_classes - 1):

            mask = labels == c

            if len(probs[mask]) > 0:
    
                boxes_to_process = boxes[mask]
                score_to_process = probs[mask]

                nms_index = tf.image.non_max_suppression(boxes_to_process, score_to_process, 300,
                                                         iou_threshold=0.4, score_threshold=self.confidence)

                nms_boxes = tf.gather(boxes_to_process, nms_index).numpy()
                nms_score = tf.gather(score_to_process, nms_index).numpy()
                nms_label = c * np.ones((len(nms_index), 1))

                if result is None:
                    result = np.concatenate((nms_label, np.expand_dims(nms_score, axis=-1), nms_boxes), axis=1)
                else:
                    result = np.vstack((result,
                                        np.concatenate((nms_label, np.expand_dims(nms_score, axis=-1), nms_boxes), axis=1)))

        top_label = result[:, 0]
        top_conf = result[:, 1]
        top_boxes = result[:, 2:]

        top_boxes[:, 0] = top_boxes[:, 0] * old_w
        top_boxes[:, 1] = top_boxes[:, 1] * old_h
        top_boxes[:, 2] = top_boxes[:, 2] * old_w
        top_boxes[:, 3] = top_boxes[:, 3] * old_h

        for i, c in enumerate(top_label):

            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

            left, top, right, bottom = top_boxes[i]

            f.write("{} {:.6} {} {} {} {}\n".format(
                predicted_class, score, str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    frcnn = Frcnn("../model/voc_1.7687.h5")

    image_infos = open("../config/test.txt").read().strip().split('\n')

    if not os.path.exists("./input"):
        os.makedirs("./input")
    if not os.path.exists("./input/detection-results"):
        os.makedirs("./input/detection-results")
    if not os.path.exists("./input/images-optional"):
        os.makedirs("./input/images-optional")

    process_bar = tqdm(range(len(image_infos)), ncols=100, unit="step")
    for i in process_bar:
        image_boxes = image_infos[i].split(' ')
        image_path = image_boxes[0]
        image_id = os.path.basename(image_path)[:-4]

        image = Image.open(image_path)
       
        frcnn.detect_single_image(image, image_id)

    print("Conversion completed!")
