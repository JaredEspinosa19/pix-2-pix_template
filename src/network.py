"""
Network Module
Define la arquitectura del Generador, Discriminador y funciones de pérdida para Pix2Pix.
"""

import tensorflow as tf


class MyLayer(tf.keras.layers.Layer):
    """Capa personalizada para redimensionar tensores."""

    def __init__(self):
        super(MyLayer, self).__init__()

    def call(self, x, y):
        """Redimensiona x al tamaño de y usando interpolación nearest neighbor."""
        new_size = tf.shape(y)[1:3]
        resized_x = tf.image.resize(x, new_size, method='nearest')
        return resized_x


def downsample(filters, size, apply_batchnorm=True):
    """
    Bloque de downsampling (encoder) del generador.

    Args:
        filters: Número de filtros de la convolución
        size: Tamaño del kernel
        apply_batchnorm: Si aplicar batch normalization

    Returns:
        Sequential model con Conv2D, BatchNorm (opcional) y LeakyReLU
    """
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters, size, strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size=(2, 2), padding='same', stride=2, apply_dropout=False):
    """
    Bloque de upsampling (decoder) del generador.

    Args:
        filters: Número de filtros de la convolución transpuesta
        size: Tamaño del kernel
        padding: Tipo de padding
        stride: Stride de la convolución
        apply_dropout: Si aplicar dropout

    Returns:
        Sequential model con Conv2DTranspose, BatchNorm, Dropout (opcional) y ReLU
    """
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters, size, strides=stride,
            padding=padding,
            kernel_initializer=initializer,
            use_bias=False
        )
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator(output_channels=1, img_height=413, img_width=1024):
    """
    Generador U-Net para Pix2Pix.

    Arquitectura encoder-decoder con skip connections.

    Args:
        output_channels: Número de canales de salida (1 para escala de grises)
        img_height: Altura de la imagen
        img_width: Ancho de la imagen

    Returns:
        Modelo de Keras del generador
    """
    inputs = tf.keras.layers.Input(shape=[img_height, img_width, output_channels])

    # Encoder (downsampling)
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch, 207, 512, 64)
        downsample(128, 4),  # (batch, 104, 256, 128)
        downsample(256, 4),  # (batch, 52, 128, 256)
        downsample(512, 4),  # (batch, 26, 64, 512)
        downsample(512, 4),  # (batch, 13, 32, 512)
        downsample(512, 4),  # (batch, 7, 16, 512)
        downsample(512, 4),  # (batch, 4, 8, 512)
        downsample(512, 4),  # (batch, 2, 4, 512)
    ]

    # Decoder (upsampling)
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch, 4, 8, 1024)
        upsample(512, 4),  # (batch, 7, 16, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch, 13, 32, 1024)
        upsample(512, 4),  # (batch, 26, 64, 1024)
        upsample(256, 4),  # (batch, 52, 128, 512)
        upsample(128, 4),  # (batch, 104, 256, 256)
        upsample(64, 4),  # (batch, 207, 512, 128)
    ]

    # Última capa del decoder
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 4,
        strides=2,
        padding='same',
        kernel_initializer=initializer,
        activation='tanh'
    )  # (batch, 413, 1024, 1)

    x = inputs

    # Aplicar encoder y guardar skip connections
    skips = []
    for i, down in enumerate(down_stack):
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Aplicar decoder con skip connections
    for i, (up, skip) in enumerate(zip(up_stack, skips)):
        x = up(x)

        # Ajustar dimensiones si no coinciden
        if x.shape != skip.shape:
            x = MyLayer()(x, skip)

        # Concatenar skip connection
        x = tf.keras.layers.Concatenate()([x, skip])

    # Aplicar última capa
    x = last(x)

    # Ajustar al tamaño de salida esperado
    tensor_random = tf.random.uniform([1, img_height, img_width, 1], minval=0, maxval=255, dtype=tf.int32)
    x = MyLayer()(x, tensor_random)

    return tf.keras.Model(inputs=inputs, outputs=x, name='Generator')


def Discriminator(img_height=413, img_width=1024):
    """
    Discriminador PatchGAN para Pix2Pix.

    Clasifica si patches de la imagen son reales o generados.

    Args:
        img_height: Altura de la imagen
        img_width: Ancho de la imagen

    Returns:
        Modelo de Keras del discriminador
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[img_height, img_width, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[img_height, img_width, 1], name='target_image')

    # Concatenar entrada y objetivo
    x = tf.keras.layers.concatenate([inp, tar])  # (batch, 413, 1024, 2)

    # Downsampling
    down1 = downsample(64, 4, False)(x)  # (batch, 207, 512, 64)
    down2 = downsample(128, 4)(down1)  # (batch, 104, 256, 128)
    down3 = downsample(256, 4)(down2)  # (batch, 52, 128, 256)
    down4 = downsample(512, 4)(down3)  # (batch, 26, 64, 512)

    # ZeroPadding y convolución
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down4)  # (batch, 28, 66, 512)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1,
        kernel_initializer=initializer,
        use_bias=False
    )(zero_pad1)  # (batch, 25, 63, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch, 27, 65, 512)

    # Capa final de decisión
    last = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer
    )(zero_pad2)  # (batch, 24, 62, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last, name='Discriminator')


class Pix2PixLoss:
    """Funciones de pérdida para el modelo Pix2Pix."""

    def __init__(self, lambda_l1=100):
        """
        Args:
            lambda_l1: Peso de la pérdida L1 en la función de pérdida del generador
        """
        self.LAMBDA = lambda_l1
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def generator_loss(self, disc_generated_output, gen_output, target):
        """
        Calcula la pérdida del generador.

        Combina:
        - Pérdida GAN: Qué tan bien engaña al discriminador
        - Pérdida L1: Qué tan cerca está de la imagen objetivo

        Args:
            disc_generated_output: Salida del discriminador para imágenes generadas
            gen_output: Imagen generada
            target: Imagen objetivo real

        Returns:
            total_gen_loss: Pérdida total del generador
            gan_loss: Componente GAN de la pérdida
            l1_loss: Componente L1 de la pérdida
        """
        # Pérdida GAN: queremos que el discriminador clasifique las generadas como reales (1)
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # Pérdida L1: diferencia absoluta entre generado y objetivo
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        # Pérdida total
        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """
        Calcula la pérdida del discriminador.

        El discriminador debe:
        - Clasificar imágenes reales como reales (1)
        - Clasificar imágenes generadas como falsas (0)

        Args:
            disc_real_output: Salida del discriminador para imágenes reales
            disc_generated_output: Salida del discriminador para imágenes generadas

        Returns:
            total_disc_loss: Pérdida total del discriminador
        """
        # Pérdida para imágenes reales
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        # Pérdida para imágenes generadas
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        # Pérdida total
        total_disc_loss = real_loss + generated_loss

        return total_disc_loss


def create_pix2pix_model(img_height=413, img_width=1024, output_channels=1, lambda_l1=100):
    """
    Crea el modelo completo Pix2Pix con generador, discriminador y funciones de pérdida.

    Args:
        img_height: Altura de las imágenes
        img_width: Ancho de las imágenes
        output_channels: Número de canales de salida
        lambda_l1: Peso de la pérdida L1

    Returns:
        generator: Modelo generador
        discriminator: Modelo discriminador
        loss_fn: Objeto con funciones de pérdida
    """
    print("\nCreando modelo Pix2Pix...")

    # Crear modelos
    generator = Generator(output_channels, img_height, img_width)
    discriminator = Discriminator(img_height, img_width)

    # Crear funciones de pérdida
    loss_fn = Pix2PixLoss(lambda_l1)

    print(f"✓ Generador creado")
    print(f"✓ Discriminador creado")
    print(f"✓ Funciones de pérdida configuradas (lambda_l1={lambda_l1})")

    return generator, discriminator, loss_fn


def create_optimizers(learning_rate=2e-4, beta_1=0.5):
    """
    Crea los optimizadores para el generador y discriminador.

    Args:
        learning_rate: Tasa de aprendizaje
        beta_1: Parámetro beta_1 de Adam

    Returns:
        generator_optimizer: Optimizador del generador
        discriminator_optimizer: Optimizador del discriminador
    """
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=beta_1)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=beta_1)

    print(f"✓ Optimizadores creados (lr={learning_rate}, beta_1={beta_1})")

    return generator_optimizer, discriminator_optimizer