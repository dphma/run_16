from tensorflow.keras import layers, models

class BasicResBlock(layers.Layer):

    def __init__(self, filters, strides=1, **kwargs):
      
        filter1, filter2, filter3 = filters
        super(BasicResBlock, self).__init__(**kwargs)

        self.conv1 = layers.Conv2D(filter1, kernel_size=1, strides=strides, use_bias=False,
                                   kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filter2, kernel_size=3, strides=strides, use_bias=False, padding='same',
                                   kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(filter3, kernel_size=1, strides=strides, use_bias=False,
                                   kernel_initializer='he_normal')
        self.bn3 = layers.BatchNormalization()

        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False, **kwargs):
        
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([inputs, x])
        x = self.relu(x)

        return x


class BottleneckResBlock(layers.Layer):

    def __init__(self, filters, strides=1, **kwargs):
       
        filter1, filter2, filter3 = filters
        super(BottleneckResBlock, self).__init__(**kwargs)

        self.shortcut = layers.Conv2D(filter3, kernel_size=1, strides=strides, use_bias=False,
                                      kernel_initializer='he_normal')
        self.shortcut_bn = layers.BatchNormalization()

        self.conv1 = layers.Conv2D(filter1, kernel_size=1, strides=strides, use_bias=False,
                                   kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filter2, kernel_size=3, use_bias=False, padding='same',
                                   kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(filter3, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()

        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False, **kwargs):
      
        x = self.shortcut(inputs)
        x = self.shortcut_bn(x, training=training)
        shortcut = self.relu(x)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([shortcut, x])
        x = self.relu(x)

        return x


class BasicResTDBlock(layers.Layer):

    def __init__(self, filters, strides=1, **kwargs):
       
        filter1, filter2, filter3 = filters
        super(BasicResTDBlock, self).__init__(**kwargs)

        self.conv1_td = layers.TimeDistributed(layers.Conv2D(filter1, kernel_size=1, strides=strides, use_bias=False,
                                                             kernel_initializer='he_normal'))
        self.bn1_td = layers.TimeDistributed(layers.BatchNormalization(axis=-1))

        self.conv2_td = layers.TimeDistributed(layers.Conv2D(filter2, kernel_size=3, strides=strides, use_bias=False, padding='same',
                                                             kernel_initializer='he_normal'))
        self.bn2_td = layers.TimeDistributed(layers.BatchNormalization(axis=-1))

        self.conv3_td = layers.TimeDistributed(layers.Conv2D(filter3, kernel_size=1, strides=strides, use_bias=False,
                                                             kernel_initializer='he_normal'))
        self.bn3_td = layers.TimeDistributed(layers.BatchNormalization(axis=-1))

        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False, **kwargs):
        
        x = self.conv1_td(inputs)
        x = self.bn1_td(x, training=training)
        x = self.relu(x)

        x = self.conv2_td(x)
        x = self.bn2_td(x, training=training)
        x = self.relu(x)

        x = self.conv3_td(x)
        x = self.bn3_td(x, training=training)

        x = self.add([inputs, x])
        x = self.relu(x)

        return x


class BottleneckResTDBlock(layers.Layer):

    def __init__(self, filters, strides=1, **kwargs):
       
        filter1, filter2, filter3 = filters
        super(BottleneckResTDBlock, self).__init__(**kwargs)

        self.shortcut_td = layers.TimeDistributed(layers.Conv2D(filter3, kernel_size=1, strides=strides, use_bias=False,
                                                                kernel_initializer='he_normal'))
        self.shortcut_bn_td = layers.TimeDistributed(layers.BatchNormalization(axis=-1))

        self.conv1_td = layers.TimeDistributed(layers.Conv2D(filter1, kernel_size=1, strides=strides, use_bias=False,
                                                             kernel_initializer='he_normal'))
        self.bn1_td = layers.TimeDistributed(layers.BatchNormalization(axis=-1))

        self.conv2_td = layers.TimeDistributed(layers.Conv2D(filter2, kernel_size=3, use_bias=False, padding='same',
                                                             kernel_initializer='he_normal'))
        self.bn2_td = layers.TimeDistributed(layers.BatchNormalization(axis=-1))

        self.conv3_td = layers.TimeDistributed(layers.Conv2D(filter3, kernel_size=1, use_bias=False,
                                                             kernel_initializer='he_normal'))
        self.bn3_td = layers.TimeDistributed(layers.BatchNormalization(axis=-1))

        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False, **kwargs):
       
        x = self.shortcut_td(inputs)
        x = self.shortcut_bn_td(x, training=training)
        shortcut = self.relu(x)

        x = self.conv1_td(inputs)
        x = self.bn1_td(x, training=training)
        x = self.relu(x)

        x = self.conv2_td(x)
        x = self.bn2_td(x, training=training)
        x = self.relu(x)

        x = self.conv3_td(x)
        x = self.bn3_td(x, training=training)

        x = self.add([shortcut, x])
        x = self.relu(x)

        return x


def make_layer(filters, num, name, strides=2):
   
    layer_list = [BottleneckResBlock(filters, strides=strides)]
    for _ in range(num):
        layer_list.append(BasicResBlock(filters))

    return models.Sequential(layer_list, name=name)


def ResNet50(input_image):
    """
    ResNet50 backbone
    """
    # input_shape(None, 600, 600, 3)
    x = layers.ZeroPadding2D((3, 3))(input_image)

    # (606, 606, 3)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, name='conv1',
                      use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1_BatchNorm")(x)
    x = layers.ReLU()(x)

    # (300, 300, 64)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # (150, 150, 64)
    x = make_layer([64, 64, 256], 2, 'conv_2x', strides=1)(x)

    # (150, 150, 256)
    x = make_layer([128, 128, 512], 3, 'conv_3x', strides=2)(x)

    # (75, 75, 512)
    feature_map = make_layer([256, 256, 1024], 5, 'conv_4x', strides=2)(x)

    # (38, 38, 512)
    return feature_map


def classifier_layers(x):
  
    x = BottleneckResTDBlock([512, 512, 2048], strides=2)(x)
    x = BasicResTDBlock([512, 512, 2048])(x)
    x = BasicResTDBlock([512, 512, 2048])(x)
    x = layers.TimeDistributed(layers.AveragePooling2D((7, 7)), name='avg_pool')(x)

    return x
