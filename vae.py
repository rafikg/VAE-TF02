import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape, \
    Conv2DTranspose

# Destroys the current TF graph
tf.keras.backend.clear_session()


class Encoder(tf.keras.layers.Layer):
    f"""
    Encoder layer with two conv layers and one dense layer
    """

    def __init__(self, latent_dim=32, name='encoder'):
        super().__init__(name=name)
        self.latent_dim = latent_dim

        self.conv1 = Conv2D(filters=32, kernel_size=3, strides=2,
                            activation='relu',
                            name='conv1')
        self.conv2 = Conv2D(filters=64, kernel_size=3, strides=2,
                            activation='relu', name='conv2')
        self.flatten = Flatten(name='flatten')
        self.dense = Dense(latent_dim * 2, name='dense')
        self.sampling = Sampling()

    def call(self, inputs, **kwargs):
        inputs = self.conv1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.flatten(inputs)
        inputs = self.dense(inputs)
        z_mean, z_log_var = tf.split(inputs, num_or_size_splits=2, axis=1)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(tf.keras.layers.Layer):
    f"""
    Decoder layer with 1 dense layer and 3 Conv2DTranspose layers
    """

    def __init__(self, name='decoder'):
        super().__init__(name=name)
        self.dense = Dense(units=7 * 7 * 32, activation='relu', name='dense')
        self.reshape = Reshape(target_shape=(7, 7, 32), name='reshape')
        self.conv_trans_1 = Conv2DTranspose(filters=64,
                                            kernel_size=3,
                                            strides=(2, 2),
                                            padding='SAME',
                                            activation='relu',
                                            name='conv_trans_1')
        self.conv_trans_2 = Conv2DTranspose(filters=32,
                                            kernel_size=3,
                                            strides=(2, 2),
                                            padding='SAME',
                                            activation='relu',
                                            name='conv_trans_2')

        # sigmoid activation (get image with origianl size)
        self.conv_trans_3 = Conv2DTranspose(filters=1,
                                            kernel_size=3,
                                            strides=(1, 1),
                                            padding='SAME',
                                            activation='sigmoid',
                                            name='conv_trans_3')

    def call(self, inputs, **kwargs):
        inputs = self.dense(inputs)
        inputs = self.reshape(inputs)
        inputs = self.conv_trans_1(inputs)
        inputs = self.conv_trans_2(inputs)
        inputs = self.conv_trans_3(inputs)
        return inputs


class Sampling(tf.keras.layers.Layer):
    f"""
    Sampling layer that implement the parametrization trick
    """

    def __init__(self, name='sampling'):
        super().__init__(name=name)

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch, dim = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    f"""
    Keras Model Subclass encalpsulate the VAE model
    """

    def __init__(self,
                 latent_dim,
                 loss_object,
                 optimizer,
                 train_loss,
                 name='VAE'):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.train_loss = train_loss
        # encoder-decoder layers
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder()

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # KL divergence Loss
        kl_loss = - 0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed

    @tf.function
    def train_for_one_step(self, train_x):
        """
        Training for one step. We decorate this function to make it
        faster.
        Parameters
        ----------
        train_x : tf.data

        Returns
        -------


        """
        with tf.GradientTape() as tape:
            image_reconst = self.__call__(train_x)
            loss = self.loss_object(train_x, image_reconst)
            loss += self.losses
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.train_loss(loss)

    def fit(self, train_data, test_data, epochs):
        """
        Train the VAE model for epochs number. This function call the train
        Parameters
        ----------
        train_data : tf.data
        test_data : np.ndarray
        epochs : int

        Returns
        -------

        """
        for epoch in range(epochs):
            for train_x in train_data:
                self.train_for_one_step(train_x)
            template = 'Epoch {}, Loss: {}'
            print(template.format(epoch + 1, self.train_loss.result()))
            self.generate_and_save_images(epoch, test_data)
            # Reset the metrics for the next epoch
            self.train_loss.reset_states()

    def generate_and_save_images(self, epoch, test_input):
        """
        Generate and save image using the random test_input
        Parameters
        ----------
        epoch : int
        test_input : np.ndarray

        Returns
        -------

        """
        predictions = self.decoder(test_input)
        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig('result/image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()


if __name__ == "__main__":

    # Define some `hyper-parameters`
    TRAIN_BUF = 1024
    BATCH_SIZE = 100
    TEST_BUF = 10000
    EPOCHS = 100
    NUMBER_OF_GENERATED_IMG = 16
    LATENT_DIM = 64
    RANDOM_VECTOR_FOR_GENERATION = tf.random.normal(
        shape=[NUMBER_OF_GENERATED_IMG, LATENT_DIM])

    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28,
                                        1).astype(
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

    # tf.data to create batches and shuffle dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    train_dataset = train_dataset.shuffle(TRAIN_BUF).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_images)
    test_dataset = test_dataset.shuffle(TEST_BUF).batch(BATCH_SIZE)

    # Specify loss object
    loss_object = tf.keras.losses.BinaryCrossentropy()

    # Specify the optimizer
    optimizer = tf.keras.optimizers.Adam(1e-3)

    # Specify metrics for training loss
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    # Create an instance of the model
    model = VAE(latent_dim=LATENT_DIM,
                loss_object=loss_object,
                optimizer=optimizer,
                train_loss=train_loss)

    model.fit(train_data=train_dataset, test_data=RANDOM_VECTOR_FOR_GENERATION,
              epochs=EPOCHS)
