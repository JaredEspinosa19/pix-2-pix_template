import tensorflow as tf
# from keras.models import load_model

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display

import numpy as np

from PIL import Image

#Variables
imagenes_d = 'canny_inv_erosion_dataset'
dataset = f'canny_inv_erosion_dataset'
generated_images_dataset = 'canny_inv_erosion'

BUFFER_SIZE = 400

BATCH_SIZE = 1

IMG_WIDTH = 1024
IMG_HEIGHT = 413
OUTPUT_CHANNELS = 1

LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                 discriminator_optimizer=discriminator_optimizer,
#                                 generator=generator,
#                                 discriminator=discriminator)
###########

def showing_example_image():
    sample_image = tf.io.read_file(str(f'{dataset}/train/SEM Imaging_ETD-SE_q1_x01_y01_s0004.png'))
    sample_image = tf.io.decode_jpeg(sample_image, channels=1)
    print(sample_image.shape)
    plt.figure()
    plt.imshow(sample_image[:, :, 0], cmap='gray')  # Utiliza cmap='gray'
    plt.axis('off')  # Opcional: elimina los ejes para una mejor visualización
    plt.show()

def load(image_file):
    # Leer y decodificar la imagen como un tensor uint8 con un canal (escala de grises)
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image, channels=1)

    # Obtener la mitad de la anchura para dividir la imagen
    w = tf.shape(image)[1]
    w = w // 2

    # Dividir la imagen en dos partes: entrada y real
    input_image = image[:, w:, :]  # Parte derecha
    real_image = image[:, :w, :]   # Parte izquierda

    # Convertir ambas imágenes a tensores float32
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

def showing_two_images():
    inp, re = load(str(f'{dataset}/train/SEM Imaging_ETD-SE_q1_x01_y01_s0004.png'))
    # Casting to int for matplotlib to display the images
    inp = inp / 255.0
    re = re / 255.0


    # Visualizar la imagen de entrada
    plt.figure()
    plt.title("Input Image")
    plt.imshow(tf.squeeze(inp), cmap='gray')  # tf.squeeze elimina la dimensión del canal
    plt.axis('off')

    # Visualizar la imagen real
    plt.figure()
    plt.title("Real Image")
    plt.imshow(tf.squeeze(re), cmap='gray')  # tf.squeeze elimina la dimensión del canal
    plt.axis('off')

    plt.show()

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 1])

    return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
    # Resizing to 286x286
    input_image, real_image = resize(input_image, real_image, int(IMG_HEIGHT*1.5),int(IMG_WIDTH*1.5))

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
    # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    # initializer = tf.keras.initializers.HeNormal()
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size=(2,2), padding='same', stride=2, apply_dropout=False):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    # initializer = tf.keras.initializers.HeNormal()
    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=stride,
                                    padding=padding,
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def call(self, x, y):
        new_size = tf.shape(y)[1:3]
        resized_x = tf.image.resize(x, new_size, method='nearest')
        return resized_x

def Generator():
    inputs = tf.keras.layers.Input(shape=[413, 1024, OUTPUT_CHANNELS])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 512, 207, 64)
        downsample(128, 4),  # (batch_size, 256, 104, 128)
        downsample(256, 4),  # (batch_size, 128, 52, 256)
        downsample(512, 4),  # (batch_size, 64, 26, 512)
        downsample(512, 4),  # (batch_size, 32, 13, 512)
        downsample(512, 4),  # (batch_size, 16, 7, 512)
        downsample(512, 4),  # (batch_size, 8, 4, 512)
        downsample(512, 4),  # (batch_size, 4, 2, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 4, 1024)
        upsample(512, 4, ),  # (batch_size, 16, 7, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 32, 13, 1024)
        upsample(512, 4),  # (batch_size, 64, 26, 1024)
        upsample(256, 4),  # (batch_size, 128, 52, 512)
        upsample(128, 4),  # (batch_size, 256, 104, 256)
        upsample(64, 4),  # (batch_size, 512, 207, 128)
    ]

    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    # initializer = tf.keras.initializers.HeNormal()
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 1024, 413, 3)

    x = inputs

    skips = []
    for i, down in enumerate(down_stack):
        x = down(x)
        # print(f"Down {i} shape:", x.shape)
        skips.append(x)

    skips = reversed(skips[:-1])

    for i, (up, skip) in enumerate(zip(up_stack, skips)):
        x = up(x)


        if x.shape != skip.shape:

            x = MyLayer()(x, skip)

        print(f"Up {i} shape:", x.shape)    # Debug print
        print(f"Skip {i} shape:", skip.shape)  # Debug print

        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    tensor_random = tf.random.uniform([1, 413, 1024, 1], minval=0, maxval=255, dtype=tf.int32)
    x = MyLayer()(x, tensor_random)


    return tf.keras.Model(inputs=inputs, outputs=x)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[413, 1024, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[413, 1024, 1], name='target_image')


    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 1024, 413, channels*2)


    down1 = downsample(64, 4, False)(x)       # (batch_size, 512, 207, 64)
    down2 = downsample(128, 4)(down1)         # (batch_size, 256, 104, 128)
    down3 = downsample(256, 4)(down2)         # (batch_size, 128, 52, 256)
    down4 = downsample(512, 4)(down3)         # (batch_size, 64, 26, 512)


    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down4)  # (batch_size, 66, 28, 512)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (batch_size, 63, 25, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 65, 27, 512)

    # Capa final para obtener la salida de decisión del discriminador
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 62, 24, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def save_prediction(prediction, dataset_name, filename='prediction.png'):
    """
    Guarda una predicción como imagen en escala de grises con dimensiones exactas.
    
    Parameters:
        prediction (tf.Tensor or np.ndarray): Tensor de predicción con dimensiones (1, H, W, 1).
        filename (str): Nombre del archivo de salida.
    """
    # Convertir a numpy si es un tensor de TensorFlow
    if isinstance(prediction, tf.Tensor):
        prediction = prediction.numpy()
    
    # Eliminar dimensiones adicionales (1, H, W, 1) -> (H, W)
    if prediction.ndim == 4 and prediction.shape[0] == 1:
        prediction = np.squeeze(prediction, axis=(0, -1))  # (H, W)
    elif prediction.ndim != 2:
        raise ValueError("Las dimensiones de entrada deben ser (1, H, W, 1) o (H, W).")
    
    # Escalar valores a [0, 1] si están en el rango [-1, 1]
    if prediction.min() < 0 or prediction.max() > 1:
        prediction = (prediction + 1) / 2.0  # Escalado de [-1, 1] a [0, 1]

    # Clip valores para garantizar que estén dentro de [0, 1]
    prediction = np.clip(prediction, 0, 1)

    # Convertir a escala de 0-255 para guardar como imagen
    prediction = (prediction * 255).astype(np.uint8)

    # Guardar la imagen usando PIL para conservar dimensiones
    img = Image.fromarray(prediction, mode='L')  # Escala de grises
    img.save(f'resultados/{dataset_name}/{filename}')
    print(f'Imagen {filename} guardada')

def generate_images(model, test_input, tar, epoch=0, name='',dataset_name='', final=False):
    prediction = model(test_input, training=True)

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    plt.figure(figsize=(15,6))

    # for i in range(3):
    #   plt.subplot(1, 3, i+1)
    #   plt.title(title[i])
    #   # Getting the pixel values in the [0, 1] range to plot.
    #   plt.imshow(display_list[i] * 0.5 + 0.5, cmap='gray')
    #   plt.axis('off')


    # if epoch % 5000 == 0:
    #   plt.savefig(f'{imagenes_d}_imagenes/epoch_{epoch}.jpg')
    #   plt.show()

    if final:
        save_prediction(prediction,dataset_name, name)

#### TRAIN ###
@tf.function
def train_step(generator, discriminator, input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

def fit(generator, discriminator, train_ds, test_ds, steps):

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            display.clear_output(wait=True)

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()

            generate_images(generator, example_input, example_target, step)
            print(f"Step: {step//1000}k")

    train_step(generator, discriminator, input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
        print('.', end='', flush=True)

    # Save (checkpoint) the model every 5k steps
    if (step + 1) % 5000 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

def main():
    print("Num GPUs disponibles: ", len(tf.config.list_physical_devices('GPU')))
    if tf.config.list_physical_devices('GPU'):
        print("TensorFlow utilizará GPU")  # Si hay GPU, se utiliza
    else:
        print("TensorFlow utilizará CPU")  # Si no hay GPU, se utiliza CPU

    # showing_example_image()
    # showing_two_images()

    train_dataset = tf.data.Dataset.list_files(f'{dataset}/train/*.png')
    train_dataset = train_dataset.map(load_image_train,
                                    num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    try:
        test_dataset = tf.data.Dataset.list_files(str(f'{dataset}/test/*.png'))
    except tf.errors.InvalidArgumentError:
        test_dataset = tf.data.Dataset.list_files(str(f'{dataset}/test/*.png'))
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    generator = Generator()
    # gen_output = generator(inp[tf.newaxis, ...], training=False)
    # plt.imshow(gen_output[0, ...]) 

    discriminator = Discriminator()
    # disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
    # plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
    # plt.colorbar()


    steps = 500000

    fit(generator, discriminator, train_dataset, test_dataset, steps=steps) 

    images_list = os.listdir(f'{dataset}/test')

    for image in images_list:
        inp, re = load_image_test(f'{dataset}/test/{image}')

        inp = tf.expand_dims(inp, axis=0)
        
        re = tf.expand_dims(re, axis=0)

        generate_images(generator, inp, re, 1000,f'{image}', dataset_name=generated_images_dataset, final=True)

    generator.save(f'{imagenes_d}_generator_gray_Normail_weights_{steps}.keras')
    discriminator.save(f'{imagenes_d}_discriminator_gray_Normal_weights_{steps}.keras')