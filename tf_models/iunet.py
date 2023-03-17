import tensorflow as tf
from conv_bn_act_block import ConvBNActBlock
from block_stack import BlockStack
from tensorflow.keras.layers import Layer
from tensorflow.nn import space_to_depth, depth_to_space
from tensorflow import concat
from tensorflow.image import resize
from tensorflow.keras import Model

class IUNet(Model):

    def __init__(self, shuffle_rate=2, in_channels=8, out_channels=3,
                 middle_channels=64, num_block=3, share_weight=False,
                 connect_mode='distinct_source', use_bn=False,
                 activation='elu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shuffle_rate = shuffle_rate
        self.connect_mode = connect_mode
        self.activation = activation

        if self.activation not in ('relu', 'leaky_relu', 'elu'):
            print('"activation" error!')
            exit(0)

        self.conv0 = ConvBNActBlock(middle_channels,
            kernel_size=3, strides=1, padding=1,
            use_bn=use_bn, activation=activation
        )
        self.block_stack = BlockStack(middle_channels,
            num_block=num_block, share_weight=share_weight, connect_mode=connect_mode,
            use_bn=use_bn, activation=activation
        )
        self.conv1 = ConvBNActBlock(out_channels * shuffle_rate ** 2,
            kernel_size=3, strides=1, padding=1,
            use_bn=False, activation=None
        )

    def call(self, image, defocus, bokeh_coarse, gamma):
        b, h, w, c = image.shape
        h_re = h // self.shuffle_rate * self.shuffle_rate
        w_re = w // self.shuffle_rate * self.shuffle_rate

        x = concat((image, defocus), axis=-1)
        x = resize(x, size=(h_re, w_re))
        x = space_to_depth(x, self.shuffle_rate)

        bokeh_coarse = tf.cond(tf.logical_or(bokeh_coarse.shape[1] != x.shape[1],
                    bokeh_coarse.shape[2] != x.shape[2]), lambda: resize(bokeh_coarse,
                                                                size=(x.shape[1],
                                                                      x.shape[2])),
                    lambda: bokeh_coarse)

        gamma = tf.ones((1, x.shape[1], x.shape[2], 1)) * 5

        x = concat((x, bokeh_coarse, gamma), axis=-1)
        x = self.conv0(x)
        x = self.block_stack(x)
        x = self.conv1(x)
        x = depth_to_space(x, self.shuffle_rate)
        x = resize(x, size=(h, w))

        return x

if __name__ == "__main__":
    from tensorflow import ones
    from pipeline import save_tflite
    from tensorflow.keras import Input
    shape = (220, 220, 3)
    data = ones((2, *shape))
    gamma = 5
    net = IUNet()
    inputs=(Input(shape=shape), Input(shape=shape), Input(shape=shape), Input(shape=(1,)))
    m = Model(inputs, outputs=(net(*inputs),))
    m.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])
    save_tflite(m, "/tmp/iunet.tflite")
