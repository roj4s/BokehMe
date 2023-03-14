from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, \
    LeakyReLU, ELU


class ConvBNActBlock(Layer):
    """!
        @class ConvBNAct
        Conv block
    """

    def __init__(self, in_channels, out_channels, kernel_size, strides, padding,
                 use_bn, activation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = Conv2D(in_channels, kernel_size, strides=strides,
                           padding=(padding, padding))

        self.bn = None
        if use_bn:
            self.bn = BatchNormalization()

        self.activation = None
        if activation == 'relu':
            self.activation = ReLU()
        elif activation == 'leaky_relu':
            self.activation = LeakyReLU()
        elif activation == 'elu':
            self.activation = ELU()


    def call(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)

        if self.activation:
            x = self.activation(x)

        return x
