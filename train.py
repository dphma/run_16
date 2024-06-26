from nets.backbone.ResNet import ResNet50
from nets import frcnn

from core import losses as losses_fn
from core.anchorGenerate import get_anchors
from core.dataReader import DataReader, get_classifier_train_data
from core.boxParse import BoundingBox

import config.config as cfg

from tensorflow.keras import optimizers, Input, models, utils
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import numpy as np
import os

class CosineAnnealSchedule(optimizers.schedules.LearningRateSchedule):
    def __init__(self, epoch, train_step, lr_max, lr_min, warmth_rate=0.2):

        super(CosineAnnealSchedule, self).__init__()

        self.total_step = epoch * train_step
        self.warm_step = self.total_step * warmth_rate
        self.lr_max = lr_max
        self.lr_min = lr_min

    @tf.function
    def __call__(self, step):
        step = tf.cast(step, tf.float32) 
        if step < self.warm_step:
            lr = self.lr_max / self.warm_step * step
        else:
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1.0 + tf.cos((step - self.warm_step) / self.total_step * np.pi)
            )

        return lr

@tf.function
def rpn_train(model, inputs, y_true):
  
    with tf.GradientTape() as tape:
        y_pred = model(inputs, training=True)
        cls_loss = losses_fn.rpn_cls_loss()(y_true[0], y_pred[0])
        regr_loss = losses_fn.rpn_regr_loss()(y_true[1], y_pred[1])

    grads = tape.gradient([cls_loss, regr_loss], model.trainable_variables)
    rpn_optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return [cls_loss, regr_loss]

@tf.function
def classifier_train(model, inputs, y_true):

    with tf.GradientTape() as tape:
        y_pred = model(inputs, training=True)
        cls_loss = losses_fn.class_loss_cls(y_true[0], y_pred[0])
        regr_loss = losses_fn.class_loss_regr(cfg.num_classes - 1)(y_true[1], y_pred[1])

    grads = tape.gradient([cls_loss, regr_loss], model.trainable_variables)
    classifier_optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return [cls_loss, regr_loss]

def main():
    global rpn_optimizer, classifier_optimizer
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    img_input = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))

    share_layer = ResNet50(img_input)
    rpn = frcnn.rpn(share_layer, num_anchors=len(cfg.anchor_box_ratios) * len(cfg.anchor_box_scales))
    classifier = frcnn.classifier(share_layer, roi_input, cfg.num_rois, nb_classes=cfg.num_classes)

    model_rpn = models.Model(img_input, rpn)
    model_classifier = models.Model([img_input, roi_input], classifier)
    model_all = models.Model([img_input, roi_input], rpn + classifier)

    anchors = get_anchors(cfg.share_layer_shape, cfg.input_shape)

    box_parse = BoundingBox(anchors, max_threshold=cfg.rpn_max_overlap, min_threshold=cfg.rpn_min_overlap)

    reader = DataReader(cfg.annotation_path, box_parse, cfg.batch_size)
    train_data = reader.generate()
    train_step = len(reader.train_lines) // cfg.batch_size

    losses = np.zeros((train_step, 4))
    best_loss = np.Inf

    rpn_lr = CosineAnnealSchedule(cfg.epoch, train_step, cfg.rpn_lr, cfg.rpn_lr * 1e-4)
    cls_lr = CosineAnnealSchedule(cfg.epoch, train_step, cfg.cls_lr, cfg.cls_lr * 1e-4)

    rpn_optimizer = optimizers.Adam(rpn_lr)
    classifier_optimizer = optimizers.Adam(cls_lr)

    for e in range(cfg.epoch):
        invalid_data = 0        
        print("Learning rate adjustment, rpn_lr: {}, cls_lr: {}".
          format(rpn_optimizer.learning_rate.numpy(),
                 classifier_optimizer.learning_rate.numpy()))

        """print("Learning rate adjustment, rpn_lr: {}, cls_lr: {}".
              format(rpn_optimizer._decayed_lr("float32").numpy(),
                     classifier_optimizer._decayed_lr("float32").numpy()))"""

    
        progbar = utils.Progbar(train_step)
        print('Epoch {}/{}'.format(e+1, cfg.epoch))
        for i in range(train_step):
            
            image, rpn_y, bbox = next(train_data)

            loss_rpn = rpn_train(model_rpn, image, rpn_y)
            predict_rpn = model_rpn(image)

            img_h, img_w = np.shape(image[0])[:2]
            
            share_layer = predict_rpn[2]
            rpn_height, rpn_width = share_layer.shape[1:-1]

            
            anchors = get_anchors(share_layer_shape=(rpn_width, rpn_height), image_shape=(img_w, img_h))
            predict_boxes = box_parse.detection_out(predict_rpn, anchors, confidence_threshold=0)

            x_roi, y_class_label, y_classifier, valid_roi = get_classifier_train_data(predict_boxes,
                                                                                      bbox,
                                                                                      img_w,
                                                                                      img_h,
                                                                                      cfg.batch_size,
                                                                                      cfg.num_classes)

            invalid_data += (cfg.batch_size - len(valid_roi))
            if len(x_roi) == 0:
                progbar.update(i+1, [('rpn_cls', np.mean(losses[:i+1, 0])),
                                     ('rpn_regr', np.mean(losses[:i+1, 1])),
                                     ('detector_cls', np.mean(losses[:i+1, 2])),
                                     ('detector_regr', np.mean(losses[:i+1, 3]))])
                continue

            loss_class = classifier_train(model_classifier,
                                          [image[valid_roi], x_roi],
                                          [y_class_label, y_classifier])

            losses[i, 0] = loss_rpn[0].numpy()
            losses[i, 1] = loss_rpn[1].numpy()
            losses[i, 2] = loss_class[0].numpy()
            losses[i, 3] = loss_class[1].numpy()

           
            progbar.update(i+1, [('rpn_cls', np.mean(losses[:i+1, 0])),
                                 ('rpn_regr', np.mean(losses[:i+1, 1])),
                                 ('detector_cls', np.mean(losses[:i+1, 2])),
                                 ('detector_regr', np.mean(losses[:i+1, 3]))])

        
        else:
            loss_rpn_cls = np.mean(losses[:, 0])
            loss_rpn_regr = np.mean(losses[:, 1])
            loss_class_cls = np.mean(losses[:, 2])
            loss_class_regr = np.mean(losses[:, 3])

            curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr

            print('\nLoss RPN classifier: {:.4f}'.format(loss_rpn_cls))
            print('Loss RPN regression: {:.4f}'.format(loss_rpn_regr))
            print('Loss Detector classifier: {:.4f}'.format(loss_class_cls))
            print('Loss Detector regression: {:.4f}'.format(loss_class_regr))
            print("{} picture can't detect any roi.".format(invalid_data))

            print('The best loss is {:.4f}. The current loss is {:.4f}.'.format(best_loss, curr_loss))
            if curr_loss < best_loss:
                best_loss = curr_loss

            print('Saving weights.\n')
            model_all.save_weights("./model/frcnn_{:.4f}.weights.h5".format(curr_loss))

if __name__ == '__main__':
    rpn_optimizer = None
    classifier_optimizer = None
    main()


