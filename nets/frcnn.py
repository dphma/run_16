from nets.backbone.ResNet import classifier_layers
from tensorflow.keras import layers
from nets.RoiPooling import RoiPooling


def rpn(share_layer, num_anchors=9):
   
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                      kernel_initializer='normal', name='rpn_conv1')(share_layer)

    x_class = layers.Conv2D(num_anchors, kernel_size=1, activation='sigmoid',
                            kernel_initializer='uniform', name='rpn_class')(x)

    x_regr = layers.Conv2D(num_anchors * 4, kernel_size=1,
                           kernel_initializer='zero', name='rpn_regress')(x)

    x_class = layers.Reshape((-1, 1), name="classification")(x_class)
    x_regr = layers.Reshape((-1, 4), name="regression")(x_regr)

    return [x_class, x_regr, share_layer]


def classifier(share_layer, input_rois, num_rois, nb_classes=21):
    
    pooling_size = 14

    out_roi_pool = RoiPooling(pooling_size, num_rois)([share_layer, input_rois])

    out = classifier_layers(out_roi_pool)
    out = layers.TimeDistributed(layers.Flatten(), name="flatten")(out)

    out_class = layers.TimeDistributed(layers.Dense(nb_classes,
                                                    activation='softmax',
                                                    kernel_initializer='zero'),
                                       name='dense_class_{}'.format(nb_classes))(out)

    out_regr = layers.TimeDistributed(layers.Dense(4 * (nb_classes-1),
                                                   activation='linear',
                                                   kernel_initializer='zero'),
                                      name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]

