from collections import OrderedDict
from ConvRNN import CLSTM_cell


# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 64, 3, 1, 1]}),

    ],

    [
        CLSTM_cell(shape=(64,64), input_channels=64, filter_size=5, num_features=64),

    ]
]

convlstm_decoder_params = [
    [

        OrderedDict({
            'conv4_leaky_1': [64, 1, 1, 1, 0]
        })
    ],

    [

        CLSTM_cell(shape=(64,64), input_channels=64, filter_size=5, num_features=64)
    ]
]

















convlstm_encoder_params = [
    [
        OrderedDict({'conv1_dropout_1_leaky_1': [1, 16, 3, 1, 1, 0.2]}),
        OrderedDict({'conv2_dropout_2_leaky_1': [64, 64, 3, 2, 1, 0.2]}),
        OrderedDict({'conv3_dropout_3_leaky_1': [96, 96, 3, 2, 1, 0.2]}),
    ],

    [
        CLSTM_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=64),
        CLSTM_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=96),
        CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96)
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_dropout_1_leaky_1': [96, 96, 4, 2, 1, 0.2]}),
        OrderedDict({'deconv2_dropout_2_leaky_1': [96, 96, 4, 2, 1, 0.2]}),
        OrderedDict({
            'conv3_dropout_3_leaky_1': [64, 16, 3, 1, 1, 0.2],
            'conv4_dropout_4_leaky_1': [16, 1, 1, 1, 0, 0.2]
        }),
    ],

    [
        CLSTM_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96),
        CLSTM_cell(shape=(32,32), input_channels=96, filter_size=5, num_features=96),
        CLSTM_cell(shape=(64,64), input_channels=96, filter_size=5, num_features=64),
    ]
]