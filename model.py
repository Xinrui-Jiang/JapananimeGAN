import gc
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras import models
from keras import losses
from keras import metrics
import numpy as np
import os
from glob import glob
from unit_layer import Flatten_Conv2D
from unit_layer import Residual_unit
from unit_layer import Down_Conv2D
from unit_layer import Up_Conv2D
from keras.initializers import he_normal
from keras_contrib.layers import InstanceNormalization
from gan_loss import gram
from gan_loss import content_loss
from gan_loss import style_loss
from gan_loss import discriminator_loss
from gan_loss import generator_loss
from tqdm import tqdm
from tensorflow.keras.applications import VGG19
from dataloader import generate_dataloader
import h5py

#build model
generator = models.Sequential()
generator.add(Flatten_Conv2D(64, (7, 7),name='Flatten_Conv2D'))
generator.add(Down_Conv2D(128, (3, 3),name='Down_Conv2D_1'))
generator.add(Down_Conv2D(256, (3, 3),name='Down_Conv2D_2'))
for i in range(1,9):
    generator.add(Residual_unit(256, (3, 3),name='Residual_unit_{}'.format(i)))
generator.add(Up_Conv2D(128, (3, 3),name='Up_Conv2D_1'))
generator.add(Up_Conv2D(64, (3, 3),name='Up_Conv2D_2'))
generator.add(Conv2D(3, (7, 7), padding='same'))

discriminator = models.Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=[256, 256, 3], kernel_initializer=he_normal()),
    LeakyReLU(0.2),
    Conv2D(64, (3, 3), strides=2, padding='same', kernel_initializer=he_normal()),
    LeakyReLU(0.2),
    Conv2D(128, (3, 3), padding='same', kernel_initializer=he_normal()),
    InstanceNormalization(),
    LeakyReLU(0.2),
    Conv2D(128, (3, 3), strides=2, padding='same', kernel_initializer=he_normal()),
    LeakyReLU(0.2),
    Conv2D(256, (3, 3), padding='same', kernel_initializer=he_normal()),
    InstanceNormalization(),
    LeakyReLU(0.2),
    Conv2D(256, (3, 3), padding='same', kernel_initializer=he_normal()),
    InstanceNormalization(),
    LeakyReLU(0.2),
    Conv2D(1, (3, 3), padding='same', kernel_initializer=he_normal())
])

gan = models.Sequential([generator, discriminator])
gan.build(input_shape=[None, 256, 256, 3])

#borrow vgg19 for computing content loss, style loss later
conv_base = VGG19(weights='imagenet',include_top=False,input_shape=(256,256,3))
auxiliary_model = models.Sequential(conv_base.layers[:-7])
auxiliary_model.add(Conv2D(512, (3, 3), activation='linear', padding='same',
                                    name='block4_conv4'))
auxiliary_model.load_weights("C:/Users/21300/.keras/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",
                             by_name=True)

#pretraining process: pretrain the generator
def pretrain_step_per_batch(real_scene_batch,generator,optimizer,metric):
    with tf.GradientTape() as tape:
        generated_imgs = generator(real_scene_batch,training=True)
        contentloss = 10*content_loss(real_scene=auxiliary_model(real_scene_batch),
                                   generated_img=auxiliary_model(generated_imgs))
    gradients = tape.gradient(contentloss,generator.trainable_variables)
    optimizer.apply_gradients(zip(gradients,generator.trainable_variables))
    metric(contentloss)

#offically start pretraining
summary_writer = tf.summary.create_file_writer("C:/Users/21300/Desktop/Japanimationgan/pretrain_filewriter")
print('Commencing generator pretraining.')

train_set = generate_dataloader(data_dir='C:/Users/21300/Desktop/Japanimationgan/Dataset/real_scene/real_scene_train')
train_set_val = generate_dataloader(data_dir='C:/Users/21300/Desktop/Japanimationgan/Dataset/real_scene/real_scene_val')

steps_per_epoch = 5229       #depending on batch size
pretrain_epochs = 2

generator_optimizer = keras.optimizers.Adam(
    learning_rate=1e-4, beta_1=0.5
)
content_loss_metric = metrics.Mean('content_loss',dtype=tf.float32)

generator.load_weights("C:/Users/21300/Desktop/Japanimationgan/weights/generator_weights.h5")

"""
reporting_step = 1743
pretrain_metric = metrics.Mean("pretrain_loss", dtype=tf.float32)
for epoch in range(pretrain_epochs):
    epoch_trained = epoch + 1   #record current process
    for step in tqdm(range(1,steps_per_epoch+1),desc=f"Pretrain Epoch {epoch + 1}/{pretrain_epochs}"):

        train_data = Image.open(str(np.asarray(train_set.next()))[3:-2])
        train_data = (np.asarray(train_data)/255.0).reshape((-1,256,256,3))

        pretrain_step_per_batch(train_data,generator,generator_optimizer,pretrain_metric)

        if step % reporting_step == 0:
            global_step = (epoch_trained-1)*steps_per_epoch + step
            with summary_writer.as_default():
                tf.summary.scalar('content_loss',content_loss_metric.result(),
                                  step=global_step)
            print('{} epoch {} steps, train content loss: '.format(epoch, step) + str(content_loss_metric.result()))
            content_loss_metric.reset_states()

            val_data = Image.open(str(np.asarray(train_set_val.next()))[3:-2])
            val_data = (np.asarray(val_data)/255.0).reshape((-1,256,256,3))

            generated_imgs = generator(val_data, training=False)
            contentloss = content_loss(real_scene=auxiliary_model(val_data),
                                            generated_img=auxiliary_model(generated_imgs))
            print('{} epoch {} steps, val content loss: '.format((epoch+1),step)+str(contentloss))
            del val_data, generated_imgs

        del train_data
        gc.collect()

        if (steps_per_epoch - step) < 5:
            val_data = Image.open(str(np.asarray(train_set_val.next()))[3:-2])
            val_data = (np.asarray(val_data) / 255.0).reshape((-1, 256, 256, 3))

            generated_imgs = np.asarray(generator(val_data, training=False)).reshape((256,256,3))
            im = Image.fromarray(np.uint8(generated_imgs * 255.))
            im.save("C:/Users/21300/Desktop/Japanimationgan/effect/pretrain/epoch{}.jpg".format(epoch)) #save some output images for validation
            del generated_imgs, val_data
            gc.collect()

generator.save_weights('C:/Users/21300/Desktop/Japanimationgan/pretrain_generator_weights.h5')
"""

#train the whole GAN architecture
def train_step_per_batch(real_scene_batch,anime_img_batch,
                         smooth_img_batch,generator,discriminator,
                         generator_optimizer,discriminator_optimizer,
                         g_loss_metric,g_adv_loss_metric,
                         content_loss_metric,style_loss_metric,
                         d_loss_metric,d_real_loss_metric,
                         d_fake_loss_metric,d_smooth_loss_metric):
    with tf.GradientTape() as g_tape,tf.GradientTape() as d_tape:
        anime_output = discriminator(anime_img_batch,training=True)
        generated_img_batch = generator(real_scene_batch,training=True)
        fake_img_output = discriminator(generated_img_batch,training=True)
        smooth_img_output = discriminator(smooth_img_batch,training=True)

        real_loss, fake_loss, smoothed_loss, d_total_loss = discriminator_loss(
            anime_output,fake_img_output,smooth_img_output
        )

        g_adv_loss = generator_loss(fake_img_output)

        vgg_gen_img = auxiliary_model(generated_img_batch)
        contentloss = content_loss(auxiliary_model(real_scene_batch),
                                   vgg_gen_img)
        styleloss = style_loss(auxiliary_model(anime_img_batch),vgg_gen_img)

        g_total_loss = g_adv_loss + contentloss + styleloss

    d_grads = d_tape.gradient(d_total_loss, discriminator.trainable_variables)
    g_grads = g_tape.gradient(g_total_loss, generator.trainable_variables)

    discriminator_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    generator_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

    g_loss_metric(g_total_loss)
    g_adv_loss_metric(g_adv_loss)
    content_loss_metric(contentloss)
    style_loss_metric(styleloss)
    d_loss_metric(d_total_loss)
    d_real_loss_metric(real_loss)
    d_fake_loss_metric(fake_loss)
    d_smooth_loss_metric(smoothed_loss)
    return 0

print("Commencing GAN training.")
gan_epochs = 5
summary_writer = tf.summary.create_file_writer("C:/Users/21300/Desktop/Japanimationgan/gan")

anime_set = generate_dataloader("C:/Users/21300/Desktop/Japanimationgan/Dataset/Anime/style_train")
smooth_set = generate_dataloader("C:/Users/21300/Desktop/Japanimationgan/Dataset/Anime/smooth_train")
anime_set_val = generate_dataloader("C:/Users/21300/Desktop/Japanimationgan/Dataset/Anime/style_val")
smooth_set_val = generate_dataloader("C:/Users/21300/Desktop/Japanimationgan/Dataset/Anime/smooth_val")

generator_optimizer = keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.5)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.5)

g_loss_metric = metrics.Mean("g_total_loss", dtype=tf.float32)
g_adv_loss_metric = metrics.Mean("g_adversarial_loss", dtype=tf.float32)
content_loss_metric = metrics.Mean("content_loss", dtype=tf.float32)
style_loss_metric = metrics.Mean("style_loss", dtype=tf.float32)
d_loss_metric = metrics.Mean("d_total_loss", dtype=tf.float32)
d_real_loss_metric = metrics.Mean("d_real_loss", dtype=tf.float32)
d_fake_loss_metric = metrics.Mean("d_fake_loss", dtype=tf.float32)
d_smooth_loss_metric = metrics.Mean("d_smooth_loss", dtype=tf.float32)

for epoch in range(gan_epochs):
    epoch_trained = epoch + 1
    steps_per_epoch = 5229
    reporting_step = 1600
    for step in tqdm(range(1,steps_per_epoch+1),desc=f"Train Epoch {epoch_trained}/{gan_epochs}"):

        train_data = Image.open(str(np.asarray(train_set.next()))[3:-2])
        train_data = (np.asarray(train_data) / 255.0).reshape((-1,256,256,3))
        anime_data = Image.open(str(np.asarray(anime_set.next()))[3:-2])
        anime_data = (np.asarray(anime_data) / 255.0).reshape((-1,256,256,3))
        smooth_data = Image.open(str(np.asarray(smooth_set.next()))[3:-2])
        smooth_data = (np.asarray(smooth_data) / 255.0).reshape((-1,256,256,3))


        train_step_per_batch(train_data,anime_data,smooth_data,
                             generator,discriminator,generator_optimizer,
                             discriminator_optimizer,
                         g_loss_metric,g_adv_loss_metric,
                         content_loss_metric,style_loss_metric,
                         d_loss_metric,d_real_loss_metric,
                         d_fake_loss_metric,d_smooth_loss_metric)

        if step % reporting_step == 0:
            global_step = (epoch_trained-1)*steps_per_epoch+step
            with summary_writer.as_default():
                tf.summary.scalar('g_total_loss',g_loss_metric.result(),step=global_step)
                tf.summary.scalar('g_adversarial_loss', g_adv_loss_metric.result(), step=global_step)
                tf.summary.scalar('content_loss', content_loss_metric.result(), step=global_step)
                tf.summary.scalar('style_loss', style_loss_metric.result(), step=global_step)
                tf.summary.scalar('d_total_loss', d_loss_metric.result(), step=global_step)
                tf.summary.scalar('d_real_loss', d_real_loss_metric.result(), step=global_step)
                tf.summary.scalar('d_fake_loss', d_fake_loss_metric.result(), step=global_step)
                tf.summary.scalar('d_smooth_loss', d_smooth_loss_metric.result(), step=global_step)

            print('{} epoch {} steps, g total loss: '.format(epoch, step) + str(g_loss_metric.result()))
            print('{} epoch {} steps, d total loss: '.format(epoch, step) + str(d_loss_metric.result()))
            g_loss_metric.reset_states()
            g_adv_loss_metric.reset_states()
            content_loss_metric.reset_states()
            style_loss_metric.reset_states()
            d_loss_metric.reset_states()
            d_real_loss_metric.reset_states()
            d_fake_loss_metric.reset_states()
            d_smooth_loss_metric.reset_states()

            #run the model on validation batch
            val_data = Image.open(str(np.asarray(train_set_val.next()))[3:-2])
            val_data = (np.asarray(val_data) / 255.0).reshape((-1,256,256,3))
            anime_val = Image.open(str(np.asarray(anime_set_val.next()))[3:-2])
            anime_val = (np.asarray(anime_val) / 255.0).reshape((-1,256,256,3))
            smooth_val = Image.open(str(np.asarray(smooth_set_val.next()))[3:-2])
            smooth_val = (np.asarray(smooth_val) / 255.0).reshape((-1,256,256,3))


            anime_output = discriminator(anime_val, training=False)
            generated_val_batch = generator(val_data, training=False)
            fake_img_output = discriminator(generated_val_batch, training=False)
            smooth_img_output = discriminator(smooth_val, training=False)

            real_loss_val, fake_loss_val, smoothed_loss_val, d_total_loss_val = discriminator_loss(
                anime_output, fake_img_output, smooth_img_output
            )

            g_adv_loss_val = losses.BinaryCrossentropy(from_logits=True)(
                tf.ones_like(fake_img_output), fake_img_output
            )

            vgg_gen_img_val = auxiliary_model(generated_val_batch)
            contentloss_val = content_loss(auxiliary_model(val_data),vgg_gen_img_val)
            styleloss_val = style_loss(auxiliary_model(anime_val), vgg_gen_img_val)

            g_total_loss_val = g_adv_loss_val + contentloss_val + styleloss_val

            print('{} epoch {} steps, d_loss: '.format(epoch,step)+str(d_total_loss_val))
            print('{} epoch {} steps, g_loss: '.format(epoch,step)+str(g_total_loss_val))

            del val_data,anime_val,smooth_val,fake_img_output,anime_output,generated_val_batch,smooth_img_output,vgg_gen_img_val
            gc.collect()

        if ((steps_per_epoch - step) < 5):
            val_data = Image.open(str(np.asarray(train_set_val.next()))[3:-2])
            val_data = (np.asarray(val_data) / 255.0).reshape((-1, 256, 256, 3))
            generated_val_batch = generator(val_data, training=False)
            generated_val_batch = np.asarray(generated_val_batch).reshape((256,256,3))
            im = Image.fromarray(np.uint8(generated_val_batch * 255.))
            im.save("C:/Users/21300/Desktop/Japanimationgan/effect/train/epoch{}_{}.jpg".format(epoch,(steps_per_epoch - step)))    #save some output images for validation
            del val_data,generated_val_batch
            gc.collect()

        del train_data,anime_data,smooth_data
        gc.collect()


generator.save_weights("C:/Users/21300/Desktop/Style_transfer_Xinrui_Jiang/1/generator_weights.h5")

gan.save_weights("C:/Users/21300/Desktop/Style_transfer_Xinrui_Jiang/1/Japanimationgan_weights.h5")
#gan.save('C:/Users/21300/Desktop/Japanimationgan/Japanimationgan.h5')


