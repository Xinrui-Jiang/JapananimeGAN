from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Add
from keras_contrib.layers import InstanceNormalization
from keras.layers import Conv2DTranspose
from keras import layers
from keras.models import Model
from keras.initializers import he_normal
from keras import models

class Flatten_Conv2D(Model):
    def __init__(self, filters, kernel_size, name, **kwargs):
        super(Flatten_Conv2D,self).__init__(name=name)
        self.model = models.Sequential()
        self.model.add(Conv2D(filters, kernel_size, padding='same', activation=layers.LeakyReLU(0.2),
                              kernel_initializer=he_normal()))
        self.model.add(InstanceNormalization())
        self.model.add(LeakyReLU(0.2))

    def build(self, input_shape):
        super(Flatten_Conv2D,self).build(input_shape)

    def call(self, x):
        return self.model(x)

class Residual_unit(Model):
    def __init__(self, filters, kernel_size, name, **kwargs):
        super(Residual_unit, self).__init__(name=name)
        self.model = models.Sequential()
        self.model.add(Conv2D(filters, kernel_size, padding='same', kernel_initializer=he_normal()))
        self.model.add(InstanceNormalization())
        self.model.add(LeakyReLU(0.2))
        self.model.add(Conv2D(filters, kernel_size, padding='same', kernel_initializer=he_normal()))
        self.model.add(InstanceNormalization())
        self.add = Add()

    def build(self, input_shape):
        super(Residual_unit, self).build(input_shape)

    def call(self, x):
        return LeakyReLU(0.2)(self.add([self.model(x), x]))

class Down_Conv2D(Model):
    def __init__(self, filters, kernel_size, name, **kwargs):
        super(Down_Conv2D, self).__init__(name=name)
        self.model = models.Sequential()
        self.model.add(Conv2D(filters, kernel_size, padding='same', strides=2, kernel_initializer=he_normal()))
        self.model.add(Conv2D(filters, kernel_size, padding='same', kernel_initializer=he_normal()))
        self.model.add(InstanceNormalization())
        self.model.add(LeakyReLU(0.2))

    def build(self, input_shape):
        super(Down_Conv2D, self).build(input_shape)

    def call(self, x):
        return self.model(x)

class Up_Conv2D(Model):
    def __init__(self, filters, kernel_size, name, **kwargs):
        super(Up_Conv2D, self).__init__(name=name)
        self.model = models.Sequential()
        self.model.add(Conv2DTranspose(filters, kernel_size, padding='same', strides=2, kernel_initializer=he_normal()))
        self.model.add(Conv2D(filters, kernel_size, padding='same', kernel_initializer=he_normal()))
        self.model.add(InstanceNormalization())
        self.model.add(LeakyReLU(0.2))

    def build(self, input_shape):
        super(Up_Conv2D, self).build(input_shape)

    def call(self, x):
        return self.model(x)
