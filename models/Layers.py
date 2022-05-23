from tensorflow.keras import layers as klayers
from utils.initial_generator import InitialGenerator
import numpy as np
from tensorflow.keras import backend as K


class STSA(klayers.Layer):

    def __init__(self, **kwargs):
        super(STSA, self).__init__(name='STSA', **kwargs)

        sin_amplitudes = InitialGenerator.initial_wave(cycle_nums=1024, duration=2048, sin_wave=True, exp=False, window='hann')
        sin_amplitudes = np.reshape(sin_amplitudes, [2048, 1, 1024])

        cos_amplitudes = InitialGenerator.initial_wave(cycle_nums=1024, duration=2048, sin_wave=False, exp=False, window='hann')
        cos_amplitudes = np.reshape(cos_amplitudes, [2048, 1, 1024])

        self.sin_conv = klayers.Conv1D(
            1024, 2048, strides=[512], padding='same', name='sin_conv1d', use_bias=False, trainable=False, weights=[sin_amplitudes]
        )
        self.sin_square = klayers.Lambda(K.square, name='sin_square')

        self.cos_conv = klayers.Conv1D(
            1024, 2048, strides=[512], padding='same', name='cos_conv1d', use_bias=False, trainable=False, weights=[cos_amplitudes]
        )
        self.cos_square = klayers.Lambda(K.square, name='cos_square')
        self.sqrt_lambda = klayers.Lambda(K.sqrt, name='sqrt')
        self.uscl_reshape = klayers.Reshape(target_shape=[63, 1024, 1], name='uscl_reshape')

    def call(self, x, **kwargs):
        x = klayers.add((self.sin_square(self.sin_conv(x)), self.cos_square(self.cos_conv(x))))
        x = self.sqrt_lambda(x)
        return self.uscl_reshape(x)


class SCNNResBlock(klayers.Layer):

    def __init__(self, filters, kernel_size, index, stride=1, conv_shortcut=True, reverse=False, **kwargs):
        super(SCNNResBlock, self).__init__(**kwargs)
        self.conv_shortcut = conv_shortcut

        i, j = kernel_size
        kernel_size_reverse = (j, i)

        if self.conv_shortcut:
            self.conv_0 = klayers.Conv2D(filters, 1, strides=stride, name=f'conv_{filters}_{index}_0')
            self.bn_0 = klayers.BatchNormalization(name=f'conv_{filters}_{index}_0_bn')

        self.relu_0 = klayers.Activation('relu', name=f'conv_{filters}_{index}_0_relu')

        self.conv_1 = klayers.Conv2D(filters, 1, strides=stride, name=f'conv_{filters}_{index}_1')
        self.bn_1 = klayers.BatchNormalization(name=f'conv_{filters}_{index}_1_bn')
        self.relu_1 = klayers.Activation('relu', name=f'conv_{filters}_{index}_1_relu')

        self.conv_2 = klayers.Conv2D(filters, kernel_size, name=f'conv_{filters}_{index}_2', padding='same')
        self.bn_2 = klayers.BatchNormalization(name=f'conv_{filters}_{index}_2_bn')
        self.relu_2 = klayers.Activation('relu', name=f'conv_{filters}_{index}_2_relu')

        self.conv_3 = klayers.Conv2D(filters, kernel_size_reverse if reverse else kernel_size, name=f'conv_{filters}_{index}_3', padding='same')
        self.bn_3 = klayers.BatchNormalization(name=f'conv_{filters}_{index}_3_bn')
        self.relu_3 = klayers.Activation('relu', name=f'conv_{filters}_{index}_3_relu')

        self.add = klayers.Add(name='conv' + str(filters) + '-' + str(index) + '_add')

    def call(self, x):
        shortcut = x
        # x = self.conv_1(x)
        # x = self.relu_1(x)
        # x = self.bn_1(x)

        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.bn_2(x)

        x = self.conv_3(x)
        x = self.relu_3(x)
        x = self.bn_3(x)

        if self.conv_shortcut:
            shortcut = self.conv_0(shortcut)
            shortcut = self.bn_0(shortcut)

        x = self.add([x, shortcut])

        return self.relu_0(x)


class USCLLayer(klayers.Layer):

    def __init__(self, filters, kernel_size, stride=1, pooling=False, padding='same', index=0, activation='relu', **kwargs):
        super(USCLLayer, self).__init__(**kwargs)
        self.pooling = pooling

        if self.pooling:
            self.pool = klayers.MaxPool2D(kernel_size)

        self.conv_1 = klayers.Conv2D(filters, kernel_size, strides=stride, name=f'uscl_{filters}_{index}_1', padding=padding)
        self.bn_1 = klayers.BatchNormalization(name=f'uscl_{filters}_{index}_1_bn')
        self.relu_1 = klayers.Activation(activation, name=f'uscl_{filters}_{index}_1_{activation}')

    def call(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.bn_1(x)

        if self.pooling:
            x = self.pool(x)

        return x
