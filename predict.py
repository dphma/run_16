from nets import frcnn
from nets.backbone.ResNet import ResNet50
import config.config as cfg
from core.anchorGenerate import get_anchors
from core.boxParse import BoundingBox
from core.dataReader import get_new_image_size

import colorsys
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from tensorflow.keras import Input, models
import tensorflow as tf


class FasterRCNN:
    def __init__(self, weight_path):
        self.class_names = cfg.label
        self.weight_path = weight_path

        self.confidence = 0.5
        
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        img_input = Input(shape=(None, None, 3))
        roi_input = Input(shape=(None, 4))
        share_layer_input = Input(shape=(None, None, 1024))

        share_layer = ResNet50(img_input)
        rpn = frcnn.rpn(share_layer, num_anchors=len(cfg.anchor_box_ratios) * len(cfg.anchor_box_scales))
        classifier = frcnn.classifier(share_layer_input, roi_input, cfg.num_rois, nb_classes=cfg.num_classes)

        self.model_rpn = models.Model(img_input, rpn)
        self.model_classifier = models.Model([share_layer_input, roi_input], classifier)

        self.model_rpn.load_weights(self.weight_path, by_name=True)
        self.model_classifier.load_weights(self.weight_path, by_name=True)

    def process_image(self, image):
      
        image_w, image_h = image.size
        new_w, new_h = get_new_image_size(image_w, image_h)

        image = image.resize((new_w, new_h), Image.BICUBIC)

        image_data = np.array(image, dtype=np.float32)
        image_data /= 255.
        image_data = np.clip(image_data, 0.0, 1.0)
        image_data = np.expand_dims(image_data, 0)  

        return image_data

    def detect_image(self, image):
        resize_image = self.process_image(image)
        old_w, old_h = image.size
        
        new_h, new_w = np.shape(resize_image)[1:-1]
        predict_rpn = self.model_rpn(resize_image, training=False)

        share_layer = predict_rpn[2]
        
        rpn_height, rpn_width = share_layer.shape[1:-1]

        anchors = get_anchors(share_layer_shape=(rpn_width, rpn_height), image_shape=(new_w, new_h))
        box_parse = BoundingBox(anchors)
        predict_boxes = box_parse.detection_out(predict_rpn, anchors, confidence_threshold=0)
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

            # p_cls, p_regr = self.model_classifier.predict([share_layer, rois])
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

        font = ImageFont.truetype(font='D:/Dung/faster-visual/run_16-main/config/simhei.ttf', size=np.floor(2e-2 * old_w + 0.5).astype('int32'))
        thickness = (old_w + old_h) // old_w * 2

        for i, c in enumerate(top_label):
            c = int(c)
            predicted_class = self.class_names[c]
            score = top_conf[i]

            
            left, top, right, bottom = top_boxes[i]
            top = top - 3
            left = left - 3
            bottom = bottom + 3
            right = right + 3

            
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(old_h, np.floor(bottom + 0.5).astype('int32'))
            right = min(old_w, np.floor(right + 0.5).astype('int32'))

            
            label = '{} {:.2f}'.format(predicted_class, score)
            print(label)
            draw = ImageDraw.Draw(image)
       
            #label_size = draw.textsize(label, font)
            label_size = font.getmask(label).getbbox()

            label = label.encode('utf-8')

            
            if top - label_size[1] >= 0:
                #text_origin = np.array([left, top - label_size[1]])
                text_width, text_height = label_size[2] - label_size[0], label_size[3] - label_size[1]
                text_origin = np.array([left, top - text_height])
            else:
                text_origin = np.array([left, top + 1])

           
            draw.rectangle([left, top, right, bottom], outline=self.colors[c], width=thickness)
            #draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple((text_origin[0] + text_width, text_origin[1] + text_height))], fill=self.colors[c])
            
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image


if __name__ == '__main__':
    img_path = 'D:/Dung/nls_12.jpg'
    faster_rcnn = FasterRCNN('D:/Dung/model/frcnn_2.0951.weights.h5')

    image = Image.open(img_path)
    image = faster_rcnn.detect_image(image)
    image.show()
