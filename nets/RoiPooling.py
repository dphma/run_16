from tensorflow.keras import layers, backend
import tensorflow as tf


class RoiPooling(layers.Layer):
    def __init__(self, pool_size, num_rois, **kwargs):

        self.pool_size = pool_size
        self.num_rois = num_rois
        super(RoiPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.channels

    def call(self, inputs, mask=None):

        assert(len(inputs) == 2)
        img = inputs[0]
        rois = inputs[1]
        outputs = []

        if img.shape[0]:        
            for b in range(img.shape[0]):
                for roi_idx in range(self.num_rois):

                    x = rois[b, roi_idx, 0]
                    y = rois[b, roi_idx, 1]
                    w = rois[b, roi_idx, 2]
                    h = rois[b, roi_idx, 3]

                    x = backend.cast(x, 'int32')
                    y = backend.cast(y, 'int32')
                    w = backend.cast(w, 'int32')
                    h = backend.cast(h, 'int32')

                    rs = tf.image.resize(img[b, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
                    
                    outputs.append(rs)

        final_output = backend.stack(outputs, axis=0)
        final_output = backend.reshape(final_output,
                                       (-1, self.num_rois, self.pool_size, self.pool_size, self.channels))

        final_output = backend.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

