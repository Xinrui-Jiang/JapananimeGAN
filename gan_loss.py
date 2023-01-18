import tensorflow as tf
from keras import losses
import tensorflow as tf
import keras
from tensorflow.keras.applications import VGG19
from keras import models
from keras import layers
from keras import losses
import tensorflow as tf
import keras
from tensorflow.keras.applications import VGG19
from keras import models
from keras import layers
from keras import losses

def gram(x):
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)

def content_loss(real_scene,generated_img):
    return 10 * losses.MeanAbsoluteError()(real_scene, generated_img)

def style_loss(anime_img, generated_img):
    anime_img = gram(anime_img)
    generated_img = gram(generated_img)
    return losses.MeanAbsoluteError()(anime_img, generated_img)

def discriminator_loss(anime_img, generated_img, smoothed_anime_img):
    loss_function = losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_function(tf.ones_like(anime_img), anime_img)
    fake_loss = loss_function(tf.zeros_like(generated_img), generated_img)
    smoothed_loss = loss_function(tf.zeros_like(smoothed_anime_img), smoothed_anime_img)
    total_loss = real_loss + fake_loss + smoothed_loss
    return real_loss, fake_loss, smoothed_loss, total_loss

def generator_loss(generated_img):
    loss_function = losses.BinaryCrossentropy(from_logits=True)
    return loss_function(tf.ones_like(generated_img), generated_img)
