# Standard Library Imports
import os
import warnings
import time
import datetime

# Third-Party Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, ReLU, Dropout, Concatenate, Multiply, ZeroPadding2D, Add
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
import cv2
from skimage import io
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

# Ignore specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# Print TensorFlow device information
print(device_lib.list_local_devices())

# Constants
BUFFER_SIZE = 133
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 1
LEARNING_RATE = 2e-4
BETA_1 = 0.5
INITIALIZER = tf.random_normal_initializer(0., 0.02)
DATA_DIR = r'C:\Users\ph1oc\Documents\DATASETS\mammalian cell segmentation cell edge noise\training_masks'
source1 = os.path.join(DATA_DIR, 'images')
source2 = os.path.join(DATA_DIR, 'masks')
LAMBDA = 100

# Function to read images and optionally apply CLAHE
def imread(in_path, apply_clahe=False):
    """Reads an image and optionally applies CLAHE."""
    img_data = io.imread(in_path)
    n_img = (255 * resize(img_data, (IMG_WIDTH, IMG_HEIGHT), mode='constant')).clip(0, 255).astype(np.uint8)
    if apply_clahe:
        clahe_tool = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        n_img = clahe_tool.apply(n_img)
    return np.expand_dims(n_img, -1)

# Load dataset for training
train_names = sorted(os.listdir(source1))
labels_names = sorted(os.listdir(source2))

training, labeling = [], []

for idx1 in tqdm(train_names):
    filepath1 = os.path.join(source1, idx1)
    if filepath1.endswith('.png'):
        file1 = mpimg.imread(filepath1)
        file1 = cv2.resize(file1, (IMG_WIDTH, IMG_HEIGHT))
        training.append(file1)

for idx2 in tqdm(labels_names):
    filepath2 = os.path.join(source2, idx2)
    if filepath2.endswith('.png'):
        file2 = mpimg.imread(filepath2)
        file2 = cv2.resize(file2, (IMG_WIDTH, IMG_HEIGHT))
        labeling.append(file2)

training = np.array(training)[:, :, :, 0] * 255
labeling = np.array(labeling)[:, :, :, 0] * 255

print('Image dataset shape: ', training.shape)
print('Mask dataset shape: ', labeling.shape)
print('------------------------------------')
print('Image dataset min/max: ', training.min(), '/', training.max())

img_vol = tf.expand_dims(training, -1)
seg_vol = tf.expand_dims(labeling, -1)

# Data augmentation configuration
dg_args = dict(
    featurewise_center=False,
    samplewise_center=False,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.01,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
    data_format='channels_last'
)

image_gen = ImageDataGenerator(**dg_args)
test_img_gen = ImageDataGenerator()

# Generator for augmented data pairs
def gen_augmented_pairs(in_vol, in_seg, batch_size=1, train=True):
    """Generates augmented image pairs."""
    while True:
        seed = np.random.choice(range(9999)) if train else 0
        g_vol = image_gen.flow(in_vol, batch_size=batch_size, seed=seed)
        g_seg = image_gen.flow(in_seg, batch_size=batch_size, seed=seed)
        for i_vol, i_seg in zip(g_vol, g_seg):
            yield i_vol, i_seg

# Prepare data generators
train_gen = gen_augmented_pairs(img_vol, seg_vol, batch_size=1)
test_gen = gen_augmented_pairs(img_vol, seg_vol, batch_size=1, train=False)

# Model definition
#####################################################################
def downsample(filters, size, apply_batchnorm=True):
    """Downsamples an input using Conv2D followed by optional BatchNormalization and LeakyReLU."""
    result = tf.keras.Sequential()
    result.add(Conv2D(filters, size, strides=2, padding='same', kernel_initializer=INITIALIZER, use_bias=False))
    if apply_batchnorm:
        result.add(BatchNormalization())
    result.add(LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    """Upsamples an input using Conv2DTranspose followed by BatchNormalization, optional Dropout, and ReLU."""
    result = tf.keras.Sequential()
    result.add(Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=INITIALIZER, use_bias=False))
    result.add(BatchNormalization())
    if apply_dropout:
        result.add(Dropout(0.5))
    result.add(ReLU())
    return result

def attention_block(x, gating_signal, inter_shape):
    """Applies an attention mechanism to the input."""
    theta_x = Conv2D(inter_shape, (1, 1), padding='same')(x)
    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(gating_signal)
    f = Add()([theta_x, phi_g])
    f = ReLU()(f)
    f = Conv2D(1, (1, 1), padding='same')(f)
    f = tf.keras.layers.Activation('sigmoid')(f)
    f = tf.image.resize(f, size=tf.shape(x)[1:3], method='nearest')
    x = Multiply()([x, f])
    return x

def Generator():
    """Builds the generator model."""
    inputs = Input(shape=[IMG_WIDTH, IMG_HEIGHT, 1])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    last = Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=INITIALIZER, activation='tanh')

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = attention_block(x, skip, 512)
        x = Concatenate()([x, skip])

    x = last(x)
    return Model(inputs=inputs, outputs=x)

# Example usage
generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

# Loss definition
#####################################################################
loss_object = BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    """Calculates the generator loss."""
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def Discriminator():
    """Builds the discriminator model."""
    inp = Input(shape=[IMG_WIDTH, IMG_HEIGHT, 1], name='input_image')
    tar = Input(shape=[IMG_WIDTH, IMG_HEIGHT, 1], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])
    down_layers = [64, 128, 256, 512, 512, 512]
    for down_filter in down_layers:
    for down_filter in down_layers:
        x = downsample(down_filter, 4, False)(x)
    zero_pad1 = ZeroPadding2D()(x)
    conv = Conv2D(512, 4, strides=1, kernel_initializer=INITIALIZER, use_bias=False)(zero_pad1)
    batchnorm1 = BatchNormalization()(conv)
    leaky_relu = LeakyReLU()(batchnorm1)
    zero_pad2 = ZeroPadding2D()(leaky_relu)
    last = Conv2D(1, 4, strides=1, kernel_initializer=INITIALIZER)(zero_pad2)
    return Model(inputs=[inp, tar], outputs=last)

# Example usage
discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

def discriminator_loss(disc_real_output, disc_generated_output):
    """Calculates the discriminator loss."""
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

# Optimizers and Checkpointing
#####################################################################
generator_optimizer = Adam(LEARNING_RATE, beta_1=BETA_1)
discriminator_optimizer = Adam(LEARNING_RATE, beta_1=BETA_1)
checkpoint_dir = './training_checkpoints_v2'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

log_dir = "logs/"
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Training function
#####################################################################
@tf.function
def train_step(input_image, target, step, metric):
    """Performs a single training step."""
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    metric.update_state(target, gen_output)

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 10)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 10)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 10)
        tf.summary.scalar('disc_loss', disc_loss, step=step // 10)

# Training loop
#####################################################################
train_accuracies = []
test_accuracies = []

def fit(train_ds, test_ds, steps):
    """Trains the model using the training dataset."""
    example_input, example_target = next(iter(test_ds))
    train_metric = tf.keras.metrics.BinaryAccuracy()
    test_metric = tf.keras.metrics.BinaryAccuracy()
    start = time.time()
    step = 0

    for input_image, target in train_ds:
        if step % 10 == 0:
            clear_output(wait=True)
            train_accuracies.append(train_metric.result().numpy())
            print(f"Step: {step}, Training Accuracy: {train_metric.result().numpy()}")
            train_metric.reset_states()
            test_metric.reset_states()
            if len(test_accuracies) == 0 or test_metric.result().numpy() > max(test_accuracies):
                checkpoint.save(file_prefix=checkpoint_prefix)

        train_step(input_image, target, step, train_metric)
        step += 1

    print(f'Time taken for {steps} steps: {time.time()-start:.2f} seconds')

# Example training call
  fit(train_gen, test_gen, steps=1000)
