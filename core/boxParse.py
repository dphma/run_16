import numpy as np
import tensorflow as tf
from core.anchorGenerate import get_anchors


class BoundingBox(object):
    def __init__(self, anchors, max_threshold=0.7, min_threshold=0.3, nms_thresh=0.7, top_k=300):
    
        self.anchors = anchors
        self.num_anchors = 0 if anchors is None else len(anchors)
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.nms_thresh = nms_thresh
        self.top_k = top_k

    def iou(self, box):
       
        inter_leftup = np.maximum(self.anchors[:, :2], box[:2])
        inter_rightdown = np.minimum(self.anchors[:, 2:4], box[2:])

        inter_wh = inter_rightdown - inter_leftup
        
        inter_wh = np.maximum(inter_wh, 0)

        
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0]) * (self.anchors[:, 3] - self.anchors[:, 1])
        
        union = area_true + area_gt - inter

        iou = inter / union

        return iou

    def ignore_box(self, box):
      
        iou = self.iou(box)

        ignored_box = np.zeros((self.num_anchors, 1))

        
        assign_mask = (iou > self.min_threshold) & (iou < self.max_threshold)

        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        
        ignored_box[:, 0][assign_mask] = iou[assign_mask]

        return ignored_box.ravel()

    def encode_box(self, box):
        
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_anchors, 5))

        assign_mask = iou > self.max_threshold

        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        
        encoded_box[:, -1][assign_mask] = iou[assign_mask]

        assigned_anchors = self.anchors[assign_mask]
        
        box_xy = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
       
        anchors_xy = 0.5 * (assigned_anchors[:, :2] + assigned_anchors[:, 2:4])
        anchors_wh = (assigned_anchors[:, 2:4] - assigned_anchors[:, :2])

        encoded_box[:, :2][assign_mask] = box_xy - anchors_xy
        encoded_box[:, :2][assign_mask] /= anchors_wh
        encoded_box[:, :2][assign_mask] *= 4

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / anchors_wh)
        encoded_box[:, 2:4][assign_mask] *= 4

        return encoded_box

    def decode_boxes(self, predictions, anchors):
 
        anchors_width = anchors[:, 2] - anchors[:, 0]
        anchors_height = anchors[:, 3] - anchors[:, 1]

        anchors_center_x = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchors_center_y = 0.5 * (anchors[:, 3] + anchors[:, 1])

        decode_bbox_center_x = predictions[:, 0] * anchors_width / 4
        decode_bbox_center_x += anchors_center_x
        decode_bbox_center_y = predictions[:, 1] * anchors_height / 4
        decode_bbox_center_y += anchors_center_y

        decode_bbox_width = np.exp(predictions[:, 2] / 4)
        decode_bbox_width *= anchors_width
        decode_bbox_height = np.exp(predictions[:, 3] / 4)
        decode_bbox_height *= anchors_height

        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)

        
        decode_bbox = np.maximum(decode_bbox, 0.0)
        decode_bbox = np.minimum(decode_bbox, 1.0)

        return decode_bbox

    def assign_boxes(self, boxes):
    
        box_data = np.zeros((self.num_anchors, 4 + 1))

        box_data[:, 4] = 0.0
        if len(boxes) == 0:
            return box_data

        ignored_boxes = np.apply_along_axis(self.ignore_box, 1, boxes[:, :4])
        
        ignored_boxes = ignored_boxes.reshape(-1, self.num_anchors, 1)
        
        ignore_iou = ignored_boxes[:, :, 0].max(axis=0)

        ignore_iou_mask = ignore_iou > 0

        
        box_data[:, 4][ignore_iou_mask] = -1

        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)

        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        
        best_iou_mask = best_iou > 0

        best_iou_idx = best_iou_idx[best_iou_mask]

        box_data[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, best_iou_mask, :4]
        box_data[:, 4][best_iou_mask] = 1

        return box_data

    def detection_out(self, predictions, anchors, confidence_threshold=0.5):
      
        p_classification = predictions[0]      
        p_regression = predictions[1]           

        pred = []

        for i in range(p_regression.shape[0]):
            
            decode_bbox = self.decode_boxes(p_regression[i], anchors)

            confidence = p_classification[i, :, 0]
            
            mask = confidence > confidence_threshold

            boxes_to_process = decode_bbox[mask]
            score_to_process = confidence[mask]

            nms_index = tf.image.non_max_suppression(boxes_to_process, score_to_process, self.top_k,
                                                     iou_threshold=self.nms_thresh)

            good_boxes = tf.gather(boxes_to_process, nms_index).numpy()
            good_score = tf.gather(score_to_process, nms_index).numpy()
            good_score = np.expand_dims(good_score, axis=1)

            predict_boxes = np.concatenate((good_score, good_boxes), axis=1)
            argsort = np.argsort(predict_boxes[:, 0])[::-1]
            predict_boxes = predict_boxes[argsort]

            pred.append(predict_boxes[:, 1:])

        return pred
