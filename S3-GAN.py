# https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/

# install keras_contrib !!!!
# pip install git+https://www.github.com/keras-team/keras-contrib.git


# example of defining composite models for training cyclegan generators
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Sequential
from keras.models import Input
from keras.layers import Conv2D, Lambda
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.layers import LeakyReLU, Reshape
from keras.layers import Dense, Embedding
from keras.initializers import RandomNormal
from keras.layers import Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.utils.vis_utils import plot_model
import os
from tqdm import tqdm
import cv2
import numpy as np
from random import randint
import random
from PIL import Image
from keras.applications.vgg19 import VGG19
import tensorflow as tf
import keras.backend as K

RES_DIR = 'res-faces-batch16-fixed'
MIXED_PATH = '%s/%d_mixed.png'
RECONSTRUCTED_A_PATH = '%s/%d_reconstructed_A.png'
RECONSTRUCTED_B_PATH = '%s/%d_reconstructed_B.png'
ORIG_A_PATH = '%s/%d_orig_A.png'
ORIG_B_PATH = '%s/%d_orig_B.png'
if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)
CONTROL_SIZE_SQRT = 3
WIDTH = 128
HEIGHT = 128
CHANNELS = 3

# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


def get_tv_loss(base_content, target):
    return tf.reduce_sum(base_content)


def gram_matrix(x):
    if len(x.shape) == 4:
        x = x[0]
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = WIDTH * HEIGHT
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    a = tf.square(
        x[:, : HEIGHT - 1, : WIDTH - 1, :] - x[:, 1:, : WIDTH - 1, :]
    )
    b = tf.square(
        x[:, : HEIGHT - 1, : WIDTH - 1, :] - x[:, : HEIGHT - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def euclidean_distance(base_content, target):
    return K.sqrt(K.sum(K.square(base_content - target), axis=-1))


def get_model():
    # Load our model. We load pretrained VGG, trained on imagenet data
    model = VGG19(include_top=False, weights='imagenet', input_shape=(WIDTH, HEIGHT, 3))
    model.trainable = False
    features_layers_content = model.get_layer('block5_conv2')

    style_outputs = [model.get_layer(name).output for name in style_layers]
    content_outputs = [model.get_layer(name).output for name in content_layers]
    model_outputs = content_outputs + style_outputs

    model2 = Model(model.input, model_outputs)

    print(model2.summary())
    return model2


# define the discriminator model
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_image = Input(shape=image_shape)
    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    # define model
    model = Model(in_image, patch_out)
    # compile model
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model


# generator a resnet block
def resnet_block(n_filters, input_layer):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # first layer convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # second convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    # concatenate merge channel-wise with input layer
    g = Concatenate()([g, input_layer])
    return g


def define_encoder(image_shape, n_resnet=3):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # c7s1-64
    g = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d128 64x64
    g = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d256 32x32
    g = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d256 16x16
    g = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d256 8x8
    g = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d256 4x4
    g = Conv2D(1024, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d256 2x2

    model = Model(in_image, g)
    print('encoder:')
    print(model.summary())
    return model


def define_decoder(latent_shape=(4, 4, 1024,), n_classes=3):
    init = RandomNormal(stddev=0.02)

    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 16384)(in_label)
    # li = Dense(2048)(li)
    print(li)
    in_latent = Input(shape=latent_shape)
    li = Reshape((4, 4, 1024))(li)
    # g = Concatenate()([in_latent, li])
    latent = Reshape((4, 4, 1024,))(in_latent)

    # 4x4
    g = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(latent)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # 8x8
    g = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # u64 16x16
    g = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # u64 32x32
    g = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # u64 32x32
    g = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # c7s1-3 64x64
    g = Conv2D(3, (7, 7), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation('tanh')(g)
    # define model
    model = Model([in_label, in_latent], out_image)
    print('decoder')
    print(model.summary())
    return model


# define the standalone generator model
def define_generator(encoder, decoder, image_shape):
    encoder.trainable = True
    decoder.trainable = True

    in_image_A = Input(shape=image_shape)
    in_image_B = Input(shape=image_shape)
    in_label = Input(shape=(1,))

    latent_A = encoder(in_image_A)
    latent_B = encoder(in_image_B)
    print(latent_A)
    reconstructed_A = decoder([in_label, latent_A])
    reconstructed_B = decoder([in_label, latent_B])

    model = Model([in_image_A, in_image_B, in_label], [reconstructed_A, reconstructed_B])
    print(model.summary())
    return model


# define the standalone generator model
def define_mixer(encoder, decoder, image_shape):
    # encoder.trainable = False
    # decoder.trainable = False

    in_image_A = Input(shape=image_shape)
    in_image_B = Input(shape=image_shape)
    in_label = Input(shape=(1,))

    latent_A = encoder(in_image_A)
    latent_B = encoder(in_image_B)

    # reconstructed_A = decoder([in_label, latent_A])
    # reconstructed_B = decoder([in_label, latent_B])

    style = Lambda(lambda x: x[:, :, :, 0:512])(latent_A)
    content = Lambda(lambda x: x[:, :, :, 512:1024])(latent_B)
    mixed = Concatenate(axis=3)([style, content])
    mixed_image = decoder([in_label, mixed])

    model = Model([in_image_A, in_image_B, in_label], [mixed_image, style, content])
    print(model.summary())
    return model


# define a composite model for updating generators by adversarial and cycle loss
def define_perceptual(g_model, d_model, encoder, decoder, mixer, vgg19, image_shape):
    d_model.trainable = False
    mixer.trainable = True
    g_model.trainable = True
    encoder.trainable = False
    vgg19.trainable = False

    in_image_A = Input(shape=image_shape)
    in_image_B = Input(shape=image_shape)
    gen_image = Input(shape=image_shape)
    in_label = Input(shape=(1,))

    reconstructed_A, reconstructed_B = g_model([in_image_A, in_image_B, in_label])
    reconstructed_latent_A = encoder(reconstructed_A)
    reconstructed_style_A = Lambda(lambda x: x[:, :, :, 0:512])(reconstructed_latent_A)
    reconstructed_content_A = Lambda(lambda x: x[:, :, :, 512:1024])(reconstructed_latent_A)

    reconstructed_latent_B = encoder(reconstructed_B)
    reconstructed_style_B = Lambda(lambda x: x[:, :, :, 0:512])(reconstructed_latent_B)
    reconstructed_content_B = Lambda(lambda x: x[:, :, :, 512:1024])(reconstructed_latent_B)

    mixed_image, style, content = mixer([reconstructed_A, reconstructed_B, in_label])

    adversarial = d_model(mixed_image)

    reconstructed_latent = encoder(mixed_image)

    reconstructed_style = Lambda(lambda x: x[:, :, :, 0:512])(reconstructed_latent)
    reconstructed_content = Lambda(lambda x: x[:, :, :, 512:1024])(reconstructed_latent)
    content_vgg, style_vgg1, style_vgg2, style_vgg3, style_vgg4, style_vgg5 = vgg19(mixed_image)

    reconstructed_A_TV = Lambda(total_variation_loss)(reconstructed_A);
    reconstructed_B_TV = Lambda(total_variation_loss)(reconstructed_B);
    reconstructed_Mixed_TV = Lambda(total_variation_loss)(mixed_image);

    model = Model([in_image_A, in_image_B, gen_image, in_label],
                  [reconstructed_A, reconstructed_B, adversarial,
                   content_vgg, style_vgg1, style_vgg2, style_vgg3, style_vgg4, style_vgg5,
                   reconstructed_A_TV, reconstructed_B_TV, reconstructed_Mixed_TV])

    # opt = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
    opt = Adam(lr=0.001)
    # compile model with weighting of least squares loss and L1 loss
    model.compile(
        loss=['mae', 'mae', 'mse', 'mse', style_loss, style_loss, style_loss, style_loss, style_loss,
              get_tv_loss, get_tv_loss, get_tv_loss],
        loss_weights=[30, 30, 1, 0.0001, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 1, 1, 1], optimizer=opt)
    print(model.summary())
    return model


# input shape
image_shape = (HEIGHT, WIDTH, 3)
encoder = define_encoder(image_shape)
decoder = define_decoder()
# generator
generator = define_generator(encoder, decoder, image_shape)
# mixer:
mixer = define_mixer(encoder, decoder, image_shape)
# discriminator:
discriminator = define_discriminator(image_shape)

vgg19 = get_model()
# perceptual:
perceptual = define_perceptual(generator, discriminator, encoder, decoder, mixer, vgg19, image_shape)


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape, label):
    # generate fake instance
    X = g_model.predict([dataset, label])
    # create 'fake' class labels (0)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# select a batch of random samples, returns images and target
def generate_real_samples_AB(dataset, n_samples, patch_shape, domains=3, classes=3):
    # choose random instances
    vid_class = np.random.randint(classes, size=1)
    vid_domain = np.random.randint(domains, size=1)
    domain = [0, 1]
    if vid_domain[0] == 1:
        domain = [1, 2]
    if vid_domain[0] == 2:
        domain = [0, 2]
    class_img = np.random.randint((dataset.shape[0] / domains) / classes, size=2)

    A = dataset[vid_class[0] * 900 + domain[0] * 300 + class_img[0]]
    B = dataset[vid_class[0] * 900 + domain[1] * 300 + class_img[1]]

    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return np.array([A]), np.array([B]), y, y


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # choose random instances
    ix = np.random.randint(dataset.shape[0], size=n_samples)
    # ix = randint(0, dataset.shape[0], size=n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return X, y


# train cyclegan models
def train(discriminator, perceptual, mixer, encoder, decoder, dataset):
    # define properties of the training run
    n_epochs, n_batch, = 10000, 16
    # determine the output square shape of the discriminator
    n_patch = discriminator.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        # X_realA, X_realB, y_realA,  y_realB = generate_real_samples_AB(trainA, n_batch, n_patch)
        X_realA, y_realA = generate_real_samples(trainB, n_batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)

        y_real_labels = np.ones((n_batch))
        # mixer
        mixed_image, style, content = mixer.predict([X_realA, X_realB, y_real_labels])
        content_A, style_A_1, style_A_2, style_A_3, style_A_4, style_A_5 = vgg19.predict(X_realA)
        content_B, style_B, _, _, _, _ = vgg19.predict(X_realB)

        y_fake = np.zeros((n_batch, n_patch, n_patch, 1))
        TV_number = np.zeros((n_batch, 1))
        # perceptual
        # update discriminator for A -> [real/fake]
        dis_loss_1 = discriminator.train_on_batch(X_realA, y_realA)
        dis_loss_2 = discriminator.train_on_batch(mixed_image, y_fake)

        weighted, loss_A, loss_B, g_loss, perc_content_loss, \
        perc_style_loss_1, perc_style_loss_2, perc_style_loss_3, perc_style_loss_4, perc_style_loss_5, \
        TV_A, TV_B, TV_Mixed = \
            perceptual.train_on_batch([X_realA, X_realB, mixed_image, y_real_labels],
                                      [X_realA, X_realB, y_realA,
                                       content_B, style_A_1, style_A_2, style_A_3, style_A_4, style_A_5, TV_number,
                                       TV_number, TV_number])

        total_style_loss = perc_style_loss_1 + perc_style_loss_2 + perc_style_loss_3 + perc_style_loss_4 + perc_style_loss_5
        total_tv_loss = TV_A + TV_B + TV_Mixed
        print(
            '>%d, %s, weighted[%.3f] reconstruction[%.3f,%.3f]  discriminator[%.3f,%.3f] perceptual[%.3f,%.3f] TV[%.3f]' %
            (i + 1, RES_DIR, weighted, loss_A, loss_B, dis_loss_1, dis_loss_2, perc_content_loss, total_style_loss,
             total_tv_loss))

        if i % 10000 == 9999:
            encoder.save('./models/' + str(i) + '_encoder.h5')
            decoder.save('./models/' + str(i) + '_decoder.h5')
            mixer.save('./models/' + str(i) + '_mixer.h5')
            generator.save('./models/' + str(i) + '_generator.h5')

        if i % 10000 == 9999:
            im = Image.fromarray(np.uint8(mixed_image[0] * 127.5 + 127.5))
            im.save(MIXED_PATH % (RES_DIR, i))

            reconstructed_A, reconstructed_B = generator.predict([X_realA, X_realB, y_real_labels])

            im = Image.fromarray(np.uint8(reconstructed_A[0] * 127.5 + 127.5))
            im.save(RECONSTRUCTED_A_PATH % (RES_DIR, i))

            im = Image.fromarray(np.uint8(reconstructed_B[0] * 127.5 + 127.5))
            im.save(RECONSTRUCTED_B_PATH % (RES_DIR, i))

            im = Image.fromarray(np.uint8(X_realA[0] * 127.5 + 127.5))
            im.save(ORIG_A_PATH % (RES_DIR, i))

            im = Image.fromarray(np.uint8(X_realB[0] * 127.5 + 127.5))
            im.save(ORIG_B_PATH % (RES_DIR, i))


# Return 2 numpy arrays of the dataset
def loadDataSet(dataApath, dataBpath, imageSize=256):
    A_images = []
    B_images = []
    A_files = []
    for root, dirs, files in os.walk(dataApath, topdown=False):
        for name in files:
            A_files.append(os.path.join(root, name))
    A_files = sorted(A_files)

    B_files = []
    for root, dirs, files in os.walk(dataBpath, topdown=False):
        for name in files:
            B_files.append(os.path.join(root, name))
    B_files = sorted(B_files)

    for img_path in tqdm(A_files):
        # img = image.load_img(img_path, target_size=(imageSize, imageSize))
        img = cv2.imread(img_path)
        img = cv2.resize(img, (imageSize, imageSize), interpolation=cv2.INTER_AREA)
        # x = image.img_to_array(img)
        A_images.append(img)

    for img_path in tqdm(B_files):
        # img = image.load_img(img_path, target_size=(imageSize, imageSize))
        img = cv2.imread(img_path)
        img = cv2.resize(img, (imageSize, imageSize), interpolation=cv2.INTER_AREA)
        # x = image.img_to_array(img)
        B_images.append(img)

    npArrayA = np.array(A_images, np.float32)
    npArrayA = (npArrayA - 127.5) / 127.5
    npArrayB = np.array(B_images, np.float32)
    npArrayB = (npArrayB - 127.5) / 127.5

    print("A dtaset shape is: " + str(npArrayA.shape))
    print("B dtaset shape is: " + str(npArrayB.shape))
    return npArrayA, npArrayB


def loadDataSetFromNpy(dataPath, imageSize=256):
    train_data = np.load(dataPath)
    npArrayA = np.array(train_data, np.float32)
    npArrayA = (npArrayA - 127.5) / 127.5
    npArrayB = npArrayA
    print("A dtaset shape is: " + str(npArrayA.shape))
    print("B dtaset shape is: " + str(npArrayB.shape))
    return npArrayA, npArrayB


# load a dataset as a list of two numpy arrays
dataset = loadDataSetFromNpy('100k_128_10000.npy')  # celebA dataset
# train models
train(discriminator, perceptual, mixer, encoder, decoder, dataset)
