#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras.datasets as dataset
import keras
import keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Dense,Conv2DTranspose,BatchNormalization,Activation,Reshape,Conv2D,Flatten,LeakyReLU
from tensorflow.keras import activations
import numpy as np
from sklearn.utils import shuffle
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import os

dataset_dir = "./datset/"
model_dir = "./model/"
test_dir = "./test/"
latent_dim = 128
img_shape=(64,64,3)
batch_size=200
epochs=100
batch_size=600
def read_data():
    filenames = os.listdir(dataset_dir)
    X = img = np.array([resize(imread(dataset_dir + file_name), (64,64),
                   anti_aliasing=True)for file_name in imgs])/255.0
    Y = np.zeros((len(X),1))
    return X,Y



class GAN:
  def __init__(self,latent_dims,img_shape):
    self.generator = self.Generator(latent_dims,img_shape[-1])
    self.discriminator = self.Discriminator(img_shape)
    self.latent_dims = latent_dims

  # ------------------------generator block-------------------------------------
  def Generator(self,input_dims,channels_img):
    inputs = Input(shape=input_dims)
    X = Dense(2*2*256)(inputs)
    X = Reshape((2,2,256))(X)
    X = self.gen_block(X,1024,4,2,'same')
    X = self.gen_block(X,512,4,2,'same')
    X = self.gen_block(X,256,4,2,'same')
    X = self.gen_block(X,128,4,2,'same')
    X = Conv2DTranspose(channels_img,4,2,'same')(X)
    X = Activation('tanh')(X)
    return keras.Model(inputs=inputs, outputs=X)
  def gen_block(self,X,filters,kernel_size,stride,padding):
    X = Conv2DTranspose(filters,kernel_size,strides=stride,padding=padding)(X)
    X =BatchNormalization()(X)
    X = Activation('relu')(X)
    return X
  #--------------------------Discriminator Block--------------------------------
  def Discriminator(self,input_dims):
    inputs = Input(shape=input_dims)
    X = self.disc_block(inputs,64,4,2,"same")
    X = self.disc_block(X,128,4,2,"same")
    X = self.disc_block(X,256,4,2,"same")
    X = self.disc_block(X,256,4,2,"same")
    X = Flatten()(X)
    X = Dense(1,activation='sigmoid')(X)
    return keras.Model(inputs=inputs, outputs=X)
  def disc_block(self,X,filters,kernel_size,stride,padding):
    X = Conv2D(filters,kernel_size,strides=stride,padding=padding)(X)
    X =BatchNormalization()(X)
    X = LeakyReLU(0.2)(X)
    return X
  #----------------------------load and save weights----------------------------
  def load_weight(self):
    try:
      self.generator.load_weights(model_dir+"generator_weights.h5")
      self.discriminator.load_weights(model_dir+"discriminator_weights.h5")
    except:
      print("unable to load weights")
  def save_weight(self):
    self.generator.save_weights(model_dir+"generator_weights.h5")
    self.discriminator.save_weights(model_dir+"discriminator_weights.h5")
    print("saved_weights")

  #-----------------------Compile networks--------------------------------------
  def compile(self,loss,d_opt,g_opt):
    self.d_opt = d_opt
    self.g_opt = g_opt
    self.loss = loss
    self.d_loss_metric = keras.metrics.Mean(name="d_loss")
    self.g_loss_metric = keras.metrics.Mean(name="g_loss")
  #---------------------------print summary-------------------------------------
  def summary(self):
    print("-------------- Generator summary --------\n")
    self.generator.summary()
    print("-------------- Discriminator summary --------\n")
    self.discriminator.summary()
  #---------------------------train NN------------------------------------------
  def train(self,real):
    batch_size = tf.shape(real)[0]
    for i in range(6):
      fake_dims = tf.random.normal(shape=(batch_size,latent_dim))
      imgs = self.generator(fake_dims)
      imgs = tf.concat([imgs,real],axis=0)
      labels = tf.concat([tf.ones((batch_size,1)),tf.zeros((batch_size,1))],axis=0)
      labels += 0.1 * tf.random.uniform(tf.shape(labels))

      with tf.GradientTape(persistent = True) as tape:
        preds = self.discriminator(imgs)
        disc_loss = self.loss(labels,preds)
      grads = tape.gradient(disc_loss, self.discriminator.trainable_weights)
      self.d_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))
    fake_dims = tf.random.normal(shape=(batch_size,latent_dim,))
    labels = tf.zeros((batch_size,1))
    with tf.GradientTape(persistent = True) as tape:
      preds = self.discriminator(self.generator(fake_dims))
      gen_loss = self.loss(labels,preds)
    grads = tape.gradient(gen_loss, self.generator.trainable_weights)
    self.g_opt.apply_gradients(zip(grads, self.generator.trainable_weights))
    return {"gen_loss":gen_loss,"disc_loss":disc_loss}
  def save_samples(self,n_sample,epoch):
    random_latent_vectors = tf.random.normal(shape=(n_sample, self.latent_dims))
    generated_images = self.generator(random_latent_vectors)
    generated_images *= 255
    generated_images.numpy()
    for i in range(n_sample):
      img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
      img.save(test_dir+"generated_img_%03d_%d.png" % (epoch, i))
    print("images saved")



if __name__=="__main__":
    dcgan = GAN(latent_dim,img_shape)
    dcgan.compile(tf.keras.losses.BinaryCrossentropy(from_logits=False),keras.optimizers.Adam(learning_rate=0.00002,beta_1=0.5),keras.optimizers.Adam(learning_rate=0.00002,beta_1=0.5))
    dcgan.load_weight()
    for ep in range(0,epochs):
      X = shuffle(X)
      n_batchs = (X.shape[0]//batch_size)+1
      idx=0
      print("epoch: "+str(ep)+"/"+str(epochs))
      for i in range(n_batchs):
        losses = dcgan.train(X[idx:idx+batch_size])
        progress = "["+"="*i+">"+"-"*(n_batchs-i-1)+"]    "+"gen loss: "+"{:.04f}  ".format(float(losses["gen_loss"]))+"  disc loss: "+"{:.04f}".format(float(losses["disc_loss"]))
        print("\r", progress, end="")
      print("")
      dcgan.save_weight()
      dcgan.save_samples(3,ep)
      K.clear_session()
