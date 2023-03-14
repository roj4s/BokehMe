from conv_bn_act_block import ConvBNActBlock
from block_stack import BlockStack
from tensorflow.keras.layers import Layer
from tensorflow.nn import space_to_depth, depth_to_space
from tensorflow import concat
from tensorflow.image import resize
from tensorflow.keras import Model

class ARNet(Model):

    def __init__(self, shuffle_rate=2, in_channels=5, out_channels=4,
                 middle_channels=128, num_block=3, share_weight=False,
                 connect_mode='distinct_source', use_bn=False,
                 activation='elu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shuffle_rate = shuffle_rate
        self.connect_mode = connect_mode
        self.activation = activation

        if self.activation not in ('relu', 'leaky_relu', 'elu'):
            print('"activation" error!')
            exit(0)

        self.conv0 = ConvBNActBlock(
            in_channels=(in_channels - 1) * shuffle_rate ** 2 + 1,
            out_channels=middle_channels,
            kernel_size=3, strides=1, padding=1,
            use_bn=use_bn, activation=activation
        )
        self.block_stack = BlockStack(
            channels=middle_channels,
            num_block=num_block, share_weight=share_weight, connect_mode=connect_mode,
            use_bn=use_bn, activation=activation
        )
        self.conv1 = ConvBNActBlock(
            in_channels=middle_channels,
            out_channels=out_channels * shuffle_rate ** 2,
            kernel_size=3, strides=1, padding=1,
            use_bn=False, activation=None
        )

    def call(self, image, defocus, gamma):
        b, h, w, c = image.shape
        h_re = h // self.shuffle_rate * self.shuffle_rate
        w_re = w // self.shuffle_rate * self.shuffle_rate

        x = concat((image, defocus), axis=0)
        x = resize(x, size=(h_re, w_re))
        x = space_to_depth(x, self.shuffle_rate)
        x = concat((x, gamma), axis=0)
        x = self.conv0(x)
        x = self.block_stack(x)
        x = self.conv1(x)
        x = self.depth_to_space(x, self.shuffle_rate)
        x = resize(x, size=(h, w))

        return x

if __name__ == "__main__":
    from tensorflow import ones
    shape = (2, 200, 200, 3)
    data = ones(shape)
    gamma = ones((2, 200, 200, 1))
    net = ARNet()
    net.call(data, data, gamma)
    print(net.summary())
