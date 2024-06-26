rpn_lr = 0.001
cls_lr = 0.001

epoch = 50

anchor_box_scales = [128, 256, 512]
anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]

batch_size = 4
rpn_stride = 16
input_shape = (300, 300)
share_layer_shape = (round(input_shape[0] / rpn_stride), round(input_shape[1] / rpn_stride))


num_rois = 128
num_regions = 256

rpn_min_overlap = 0.3
rpn_max_overlap = 0.7
classifier_min_overlap = 0.1
classifier_max_overlap = 0.5
classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

data_pretreatment = 'random'

annotation_path = '/kaggle/working/new_annotation.txt'
label = ['nlb', 'nls', 'gls']

num_classes = len(label) + 1
