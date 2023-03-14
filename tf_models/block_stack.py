from conv_bn_act_block import ConvBNActBlock
from tensorflow.keras.layers import Layer
from tensorflow.keras import Input, Sequential

class BlockStack(Layer):

    def __init__(self, channels, num_block, share_weight, connect_mode, use_bn,
                 activation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = []

        for i in range(num_block):
            block = None
            if share_weight:
                block = Sequential([ConvBNActBlock(channels, channels,
                                                   kernel_size=3, strides=1,
                                                   padding=1, use_bn=use_bn,
                                                   activation=activation),
                                    ConvBNActBlock(in_channels=channels,
                                                   out_channels=channels,
                                                   kernel_size=3, strides=1,
                                                   padding=1, use_bn=use_bn,
                                                   activation=activation)])
            else:
                block = Sequential([ConvBNActBlock(in_channels=channels,
                                                   out_channels=channels,
                                                   kernel_size=3, strides=1,
                                                   padding=1, use_bn=use_bn,
                                                   activation=activation),
                                    ConvBNActBlock(in_channels=channels,
                                                   out_channels=channels,
                                                   kernel_size=3, strides=1,
                                                   padding=1, use_bn=use_bn,
                                                   activation=activation)])

            self.blocks.append(block)

        self.connect_mode = connect_mode
        self.num_block = num_block

    def call(self, x):
        for i in range(self.num_block):
            if self.connect_mode == 'no':
                x = self.blocks[i](x)

            elif self.connect_mode == 'distinct_source':
                x = self.blocks[i](x) + x

            elif self.connect_mode == 'shared_source':
                x0 = Input()(x)
                x = self.blocks[i](x) + x0

        return x
