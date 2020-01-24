import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from vae import VAE
import time
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
    'float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype(
    'float32')

# Normalizing the images
train_images /= 255.
test_images /= 255.

# Binarization
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.

TRAIN_BUF = 60000
BATCH_SIZE = 100
TEST_BUF = 10000
EPOCHS = 2
# tf.data to create batches and shuffle dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(
    TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(
    TEST_BUF).batch(BATCH_SIZE)

latent_dim = 64

vae_model = VAE(latent_dim)
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    for train_x in train_dataset:
        vae_model.compute_apply_gradients(train_x, opt)
    end_time = time.time()

    if epoch % 1 == 0:
        loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(vae_model.compute_loss(test_x))
    elbo = -loss.result()

    print('Epoch: {}, Test set ELBO: {}, '
          'time elapse for current epoch {}'.format(epoch,
                                                    elbo,
                                                    end_time - start_time))

