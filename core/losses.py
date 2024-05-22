from tensorflow.keras import backend, losses
import tensorflow as tf
import numpy as np


def rpn_cls_loss(ratio=5):
    def cls_loss(y_true, y_pred):
        
        label_true = y_true[:, :, -1]       
        label_pred = y_pred

        indices_for_object = tf.where(backend.equal(label_true, 1))     
        labels_for_object = tf.gather_nd(y_true, indices_for_object)    
        classification_for_object = tf.gather_nd(label_pred, indices_for_object)    

        cls_loss_for_object = backend.binary_crossentropy(labels_for_object, classification_for_object)

        
        indices_for_back = tf.where(backend.equal(label_true, 0))
        labels_for_back = tf.gather_nd(y_true, indices_for_back)
        classification_for_back = tf.gather_nd(label_pred, indices_for_back)

        
        cls_loss_for_back = backend.binary_crossentropy(labels_for_back, classification_for_back)

        
        normalizer_pos = tf.where(backend.equal(label_true, 1))
        normalizer_pos = backend.cast(backend.shape(normalizer_pos)[0], 'float32')
        normalizer_pos = backend.maximum(backend.cast_to_floatx(1.0), normalizer_pos)

        
        normalizer_neg = tf.where(backend.equal(label_true, 0))
        normalizer_neg = backend.cast(backend.shape(normalizer_neg)[0], 'float32')
        normalizer_neg = backend.maximum(backend.cast_to_floatx(1.0), normalizer_neg)

        
        cls_loss_for_object = backend.sum(cls_loss_for_object) / normalizer_pos         
        cls_loss_for_back = ratio * backend.sum(cls_loss_for_back) / normalizer_neg    

        loss = cls_loss_for_object + cls_loss_for_back

        return loss

    return cls_loss


def rpn_regr_loss(sigma=1.0):
    sigma_squared = sigma ** 2

    def smooth_l1(y_true, y_pred):
     
        regression_pred = y_pred
        regression_true = y_true[:, :, :-1]  
        label_true = y_true[:, :, -1]         

        indices = tf.where(backend.equal(label_true, 1))                    
        regression_pred = tf.gather_nd(regression_pred, indices)            
        regression_true = tf.gather_nd(regression_true, indices)            

        regression_diff = backend.abs(regression_pred - regression_true)
        regression_loss = tf.where(  
            backend.less(regression_diff, 1.0 / sigma_squared),             
            0.5 * sigma_squared * backend.pow(regression_diff, 2),         
            regression_diff - 0.5 / sigma_squared                          
        )

        normalizer = backend.maximum(1, backend.shape(indices)[0])
        normalizer = backend.cast(normalizer, dtype='float32')
        loss = backend.sum(regression_loss) / normalizer

        return loss

    return smooth_l1


def class_loss_regr(num_classes):
    epsilon = 1e-4

    def class_loss_regr_fixed_num(y_true, y_pred):

        regr_loss = 0
        batch_size = len(y_true)
        for i in range(batch_size):
            x = y_true[i, :, 4 * num_classes:] - y_pred[i, :, :]                    
            x_abs = backend.abs(x)                                                  
            x_bool = backend.cast(backend.less_equal(x_abs, 1.0), 'float32')        

            loss = 4 * backend.sum(
                y_true[i, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / backend.sum(
                epsilon + y_true[i, :, :4 * num_classes])
            regr_loss += loss

        return regr_loss / backend.constant(batch_size)

    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    return backend.mean(losses.categorical_crossentropy(y_true, y_pred))
