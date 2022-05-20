# DL laboratory (9/10)

### Prof. Stefano Carrazza

**Summary:** Generative models.

## Exercise: Deep convolutional GAN

1. Load the MNIST dataset and keep only the training images. Reshape images as
   [60000, 28, 28, 1], cast to `float32` and normalize images to [-1, 1].

2. Build a dataset using shuffle 60k and batch size 256 using
   `tf.data.Dataset.from_tensor_slices` with `shuffle` and `batch` methods
   respectively. Plot 25 image samples.

3. Define a generator model with the following architecture:
   - an initial dense layer with `(7, 7, 256)` units, no bias and input shape 100.
   - a batch normalization layer (`tf.keras.layers.BatchNormalization`) which
     maintains the mean output close to 0 and the output standard deviation
     close to 1.
   - a LeakyReLU layer (less prone to get stuck when the argument is negative).
   - a reshape to `(7, 7, 256)` using `tf.keras.layer.Reshape`.
   - a deconvolution layer with 128 filters, kernel size (5,5), strides (1,1),
     padding "same" and no bias (`tf.keras.layers.Conv2DTranspose`), followed by a batch normalization and leaky relu layers.
   - a deconvolution layer with 64 filters, kernel size (5,5), strides (2,2),
     padding "same" and no bias (`tf.keras.layers.Conv2DTranspose`), followed by a batch normalization and leaky relu layers.
   - a deconvolution layer with 1 filters, kernel size (5,5), strides (2,2),
     padding "same" and no bias (`tf.keras.layers.Conv2DTranspose`) and activation `tanh`.

4. Define a noise distribution of 100 points for 25 images using
   `tf.random.normal([25, 100])`. Plot 25 samples of generated images using the
   the untrained generator defined in the previous point.

5. Define a discriminator model with the following architecture:
   - a convolution layer with 64 units, kernel size (5,5), strides (2,2),
     padding "same" and input shape [28, 28, 1], followed by a leaky relu and
     dropout layer with 0.3 probability.
   - a convolution layer with 128 units, kernel size (5,5), strides (2,2), and
     padding "same", followed by a leaky relu and dropout layer with 0.3
     probability.
   - a flatten layer
   - a final dense layer with just one unit and linear activation function.

6. Define the loss functions for the generator and discriminator using the cross
   entropy (`from_logits=True`). The discriminator should distinguish real
   images from fakes. Its loss function should compare the discriminator's
   predictions on real images to an array of 1s, and the discriminator's
   predictions on fake (generated) images to an array of 0s. Similarly, the
   generator's loss quantifies how well it was able to trick the discriminator.
   Intuitively, if the generator is performing well, the discriminator will
   classify the fake images as real (or 1). Here, we should compare the
   discriminators decisions on the generated images to an array of 1s.

7. Train both models simultaneously with Adam. Print to disk samples of
   generated images at multiple epochs.





