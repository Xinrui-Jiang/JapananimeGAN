from unit_layer import Flatten_Conv2D
from unit_layer import Residual_unit
from unit_layer import Down_Conv2D
from unit_layer import Up_Conv2D
from keras import models
from keras.layers import Conv2D
from  tqdm import tqdm
from glob import glob
import numpy as np
from PIL import Image
import gc
import h5py
from keras.initializers import he_normal
from keras_contrib.layers import InstanceNormalization

generator = models.Sequential()
generator.add(Flatten_Conv2D(64, (7, 7),name='Flatten_Conv2D'))
generator.add(Down_Conv2D(128, (3, 3),name='Down_Conv2D_1'))
generator.add(Down_Conv2D(256, (3, 3),name='Down_Conv2D_2'))
for i in range(1,9):
    generator.add(Residual_unit(256, (3, 3),name='Residual_unit_{}'.format(i)))
generator.add(Up_Conv2D(128, (3, 3),name='Up_Conv2D_1'))
generator.add(Up_Conv2D(64, (3, 3),name='Up_Conv2D_2'))
generator.add(Conv2D(3, (7, 7), padding='same'))
generator.build(input_shape=[None,1080,1920,3])
generator.load_weights('C:/Users/21300/Desktop/Japanimationgan/generator_weights.h5')
generator.summary()

#test the model effects
test_dir = 'C:/Users/21300/Desktop/Japanimationgan/pred/origin'
X_pred = []

for img_path in tqdm(glob(test_dir + '/*.jpg')):
    X_pred.append(img_path)

for i,picture_path in enumerate(X_pred):
    picture = Image.open(picture_path)
    picture = np.asarray(picture) / 255.0
    picture = picture.reshape((-1,1080,1920,3))
    generated_img = np.asarray(generator(picture, training=False)).reshape((1080,1920,3))

    im = Image.fromarray(np.uint8(generated_img * 255.))
    im.save('C:/Users/21300/Desktop/Japanimationgan/pred/style/{}.jpg'.format(i+1))
    del picture,generated_img
    gc.collect()
