"""
Training Module
Maneja el entrenamiento del modelo, evaluación de métricas y guardado de checkpoints.
"""

import tensorflow as tf
import os
import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
from PIL import Image


class Pix2PixTrainer:
    """Clase para entrenar el modelo Pix2Pix."""

    def __init__(self, generator, discriminator, loss_fn,
                 generator_optimizer, discriminator_optimizer,
                 dataset_name, checkpoint_dir='./training_checkpoints',
                 log_dir='./logs'):
        """
        Args:
            generator: Modelo generador
            discriminator: Modelo discriminador
            loss_fn: Objeto con funciones de pérdida
            generator_optimizer: Optimizador del generador
            discriminator_optimizer: Optimizador del discriminador
            dataset_name: Nombre del dataset (para guardar resultados)
            checkpoint_dir: Directorio para guardar checkpoints
            log_dir: Directorio para logs de TensorBoard
        """
        self.generator = generator
        self.discriminator = discriminator
        self.loss_fn = loss_fn
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.dataset_name = dataset_name

        # Estructura de carpetas para el modelo
        self.model_base_dir = f'resultados/{dataset_name}'
        self.train_images_dir = os.path.join(self.model_base_dir, 'imagenes_entrenamiento')
        self.test_images_dir = os.path.join(self.model_base_dir, 'imagenes_prueba')
        self.weights_dir = os.path.join(self.model_base_dir, 'weights')
        self.metrics_dir = os.path.join(self.model_base_dir, 'metricas')

        # Configurar checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator
        )

        # Configurar TensorBoard
        self.log_dir = log_dir
        self.summary_writer = tf.summary.create_file_writer(
            log_dir + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        # Crear directorios si no existen
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.train_images_dir, exist_ok=True)
        os.makedirs(self.test_images_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        print(f"\n✓ Trainer inicializado")
        print(f"  - Checkpoints: {checkpoint_dir}")
        print(f"  - Logs: {log_dir}")
        print(f"  - Resultados: {self.model_base_dir}")
        print(f"    - Imágenes entrenamiento: {self.train_images_dir}")
        print(f"    - Imágenes prueba: {self.test_images_dir}")
        print(f"    - Weights: {self.weights_dir}")
        print(f"    - Métricas: {self.metrics_dir}")

    @tf.function
    def train_step(self, input_image, target, step):
        """
        Ejecuta un paso de entrenamiento.

        Args:
            input_image: Imagen de entrada
            target: Imagen objetivo
            step: Número del paso actual
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Forward pass del generador
            gen_output = self.generator(input_image, training=True)

            # Forward pass del discriminador
            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            # Calcular pérdidas
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.loss_fn.generator_loss(
                disc_generated_output, gen_output, target
            )
            disc_loss = self.loss_fn.discriminator_loss(disc_real_output, disc_generated_output)

        # Calcular gradientes
        generator_gradients = gen_tape.gradient(
            gen_total_loss,
            self.generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss,
            self.discriminator.trainable_variables
        )

        # Aplicar gradientes
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        # Registrar métricas en TensorBoard
        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)

    def generate_images(self, test_input, tar, epoch=0, show_images=False):
        """
        Genera imágenes de prueba durante el entrenamiento.

        Args:
            test_input: Imagen de entrada de prueba
            tar: Imagen objetivo de prueba
            epoch: Época actual
            show_images: Si mostrar las imágenes
        """
        prediction = self.generator(test_input, training=True)

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        if show_images and epoch % 100000 == 0:
            plt.figure(figsize=(15, 6))
            for i in range(3):
                plt.subplot(1, 3, i + 1)
                plt.title(title[i])
                plt.imshow(display_list[i] * 0.5 + 0.5, cmap='gray')
                plt.axis('off')

            plt.savefig(os.path.join(self.train_images_dir, f'epoch_{epoch}.jpg'))
            plt.show()

    def evaluate_model(self, test_dataset):
        """
        Evalúa el modelo calculando PSNR y SSIM en el dataset de prueba.

        Args:
            test_dataset: Dataset de prueba (debe estar batcheado)

        Returns:
            mean_psnr: PSNR promedio
            mean_ssim: SSIM promedio
        """
        psnr_values = []
        ssim_values = []

        for input_image, target in test_dataset:
            # input_image y target ya vienen con dimensión de batch desde el dataset
            # Forma esperada: (batch_size, height, width, channels)
            prediction = self.generator(input_image, training=False)

            # Calcular métricas (max_val=2.0 porque imágenes están en rango [-1, 1])
            psnr = tf.image.psnr(target, prediction, max_val=2.0).numpy()
            ssim = tf.image.ssim(target, prediction, max_val=2.0).numpy()

            # Extraer valores del batch
            psnr_values.append(psnr[0])
            ssim_values.append(ssim[0])

        mean_psnr = np.mean(psnr_values)
        mean_ssim = np.mean(ssim_values)

        return mean_psnr, mean_ssim

    def save_prediction(self, prediction, filename='prediction.png'):
        """
        Guarda una predicción como imagen.

        Args:
            prediction: Tensor de predicción
            filename: Nombre del archivo de salida
        """
        # Convertir a numpy si es tensor
        if isinstance(prediction, tf.Tensor):
            prediction = prediction.numpy()

        # Eliminar dimensiones extras (1, H, W, 1) -> (H, W)
        if prediction.ndim == 4 and prediction.shape[0] == 1:
            prediction = np.squeeze(prediction, axis=(0, -1))

        # Escalar de [-1, 1] a [0, 1]
        prediction = (prediction + 1) / 2.0

        # Clip valores
        prediction = np.clip(prediction, 0, 1)

        # Convertir a 0-255
        prediction = (prediction * 255).astype(np.uint8)

        # Guardar imagen
        img = Image.fromarray(prediction, mode='L')
        img.save(os.path.join(self.test_images_dir, filename))

    def fit(self, train_dataset, test_dataset, steps, eval_interval=1000, save_interval=5000):
        """
        Entrena el modelo.

        Args:
            train_dataset: Dataset de entrenamiento
            test_dataset: Dataset de prueba
            steps: Número total de pasos de entrenamiento
            eval_interval: Intervalo para evaluar el modelo
            save_interval: Intervalo para guardar checkpoints
        """
        best_score = -1
        example_input, example_target = next(iter(test_dataset.take(1)))
        start = time.time()

        print(f"\n{'='*60}")
        print(f"Iniciando entrenamiento: {steps} pasos")
        print(f"{'='*60}\n")

        for step, (input_image, target) in train_dataset.repeat().take(steps).enumerate():
            # Mostrar progreso cada 1000 pasos
            if (step) % 1000 == 0:
                display.clear_output(wait=True)

                if step != 0:
                    elapsed = time.time() - start
                    print(f'Tiempo para 1000 pasos: {elapsed:.2f} seg\n')

                start = time.time()

                self.generate_images(example_input, example_target, step)
                print(f"Paso: {step // 1000}k / {steps // 1000}k")

            # Entrenar un paso
            self.train_step(input_image, target, step)

            # Mostrar progreso
            if (step + 1) % 10 == 0:
                print('.', end='', flush=True)

            # Evaluar modelo
            if (step) % eval_interval == 0 and step != 0:
                print("\n\nEvaluando modelo...")
                psnr, ssim = self.evaluate_model(test_dataset)
                current_score = psnr + ssim

                print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")

                # Guardar mejores pesos
                if current_score > best_score:
                    best_score = current_score
                    weights_path = os.path.join(self.weights_dir, "best_generator_weights.h5")
                    self.generator.save_weights(weights_path)
                    print(f"✓ Mejores pesos guardados: {weights_path}")

                    # Guardar métricas
                    metrics_file = os.path.join(self.metrics_dir, f"metrics_step_{step}.txt")
                    with open(metrics_file, 'w') as f:
                        f.write(f"Step: {step}\n")
                        f.write(f"PSNR: {psnr:.4f}\n")
                        f.write(f"SSIM: {ssim:.4f}\n")
                        f.write(f"Score: {current_score:.4f}\n")

            # Guardar checkpoint
            if (step + 1) % save_interval == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                print(f"\n✓ Checkpoint guardado en paso {step + 1}")

        # Guardar modelos finales
        final_gen_path = os.path.join(self.weights_dir, "generator_final.keras")
        final_disc_path = os.path.join(self.weights_dir, "discriminator_final.keras")
        self.generator.save(final_gen_path)
        self.discriminator.save(final_disc_path)

        print(f"\n{'='*60}")
        print(f"Entrenamiento completado")
        print(f"{'='*60}")
        print(f"✓ Modelo generador final guardado: {final_gen_path}")
        print(f"✓ Modelo discriminador final guardado: {final_disc_path}")
        print(f"{'='*60}\n")

    def restore_checkpoint(self):
        """Restaura el último checkpoint guardado."""
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            self.checkpoint.restore(latest_checkpoint)
            print(f"✓ Checkpoint restaurado: {latest_checkpoint}")
            return True
        else:
            print("No se encontraron checkpoints previos")
            return False


class Pix2PixInference:
    """Clase para realizar inferencia con modelos entrenados."""

    def __init__(self, generator, dataset_loader):
        """
        Args:
            generator: Modelo generador entrenado
            dataset_loader: Instancia de DatasetLoader
        """
        self.generator = generator
        self.dataset_loader = dataset_loader

    def load_weights(self, weights_path):
        """
        Carga pesos del generador.

        Args:
            weights_path: Ruta al archivo de pesos
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No se encontró el archivo de pesos: {weights_path}")

        self.generator.load_weights(weights_path)
        print(f"✓ Pesos cargados: {weights_path}")

    def predict_image(self, image_path):
        """
        Realiza predicción sobre una imagen.

        Args:
            image_path: Ruta a la imagen de entrada

        Returns:
            input_image: Imagen de entrada
            prediction: Imagen predicha
        """
        # Cargar imagen directamente (sin ground truth)
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, channels=1)
        image = tf.cast(image, tf.float32)

        # Redimensionar si es necesario
        image = tf.image.resize(
            image,
            [self.dataset_loader.IMG_HEIGHT, self.dataset_loader.IMG_WIDTH],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        # Normalizar
        image = (image / 127.5) - 1

        # Expandir dimensiones para batch
        input_image_batch = tf.expand_dims(image, axis=0)

        # Realizar predicción
        prediction = self.generator(input_image_batch, training=False)

        return image, prediction[0]

    def predict_dataset(self, dataset_path, output_dir):
        """
        Realiza predicciones sobre todas las imágenes de un directorio.

        Args:
            dataset_path: Ruta al directorio con imágenes
            output_dir: Directorio donde guardar las predicciones
        """
        os.makedirs(output_dir, exist_ok=True)

        image_files = [f for f in os.listdir(dataset_path) if f.endswith('.png')]

        print(f"\nRealizando predicciones sobre {len(image_files)} imágenes...")

        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(dataset_path, image_file)

            # Predecir
            input_image, prediction = self.predict_image(image_path)

            # Guardar predicción
            self._save_prediction(prediction, os.path.join(output_dir, image_file))

            if i % 10 == 0:
                print(f"Procesadas: {i}/{len(image_files)}")

        print(f"✓ Predicciones guardadas en: {output_dir}")

    def _save_prediction(self, prediction, output_path):
        """Guarda una predicción como imagen."""
        # Convertir a numpy
        if isinstance(prediction, tf.Tensor):
            prediction = prediction.numpy()

        # Escalar de [-1, 1] a [0, 1]
        prediction = (prediction + 1) / 2.0
        prediction = np.clip(prediction, 0, 1)

        # Convertir a 0-255
        if prediction.ndim == 3:
            prediction = np.squeeze(prediction, axis=-1)

        prediction = (prediction * 255).astype(np.uint8)

        # Guardar
        img = Image.fromarray(prediction, mode='L')
        img.save(output_path)

    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualiza una predicción junto con la imagen de entrada.

        Args:
            image_path: Ruta a la imagen
            save_path: Ruta opcional para guardar la visualización
        """
        input_image, prediction = self.predict_image(image_path)

        # Escalar para visualización
        input_vis = (input_image + 1) / 2.0
        pred_vis = (prediction + 1) / 2.0

        # Crear figura
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title('Input Image')
        plt.imshow(tf.squeeze(input_vis), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Predicted Image')
        plt.imshow(tf.squeeze(pred_vis), cmap='gray')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path)
            print(f"✓ Visualización guardada: {save_path}")

        plt.show()


def calculate_metrics(generator, test_dataset, image_names=None, metrics_dir=None):
    """
    Calcula métricas PSNR y SSIM para todo el dataset de prueba.

    Args:
        generator: Modelo generador
        test_dataset: Dataset de prueba (debe estar batcheado)
        image_names: Lista opcional de nombres de archivos correspondientes a cada imagen
        metrics_dir: Directorio opcional donde guardar métricas individuales

    Returns:
        Dictionary con métricas promedio y por imagen
    """
    psnr_values = []
    ssim_values = []
    individual_metrics = []

    print("\nCalculando métricas...")

    for i, (input_image, target) in enumerate(test_dataset):
        # input_image y target ya vienen con dimensión de batch
        prediction = generator(input_image, training=False)

        # Calcular métricas (max_val=2.0 porque imágenes están en rango [-1, 1])
        psnr = tf.image.psnr(target, prediction, max_val=2.0).numpy()
        ssim = tf.image.ssim(target, prediction, max_val=2.0).numpy()

        # Extraer valores del batch
        psnr_val = psnr[0]
        ssim_val = ssim[0]
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)

        # Guardar información individual
        image_name = image_names[i] if image_names and i < len(image_names) else f"image_{i:04d}"
        individual_metrics.append({
            'image': image_name,
            'psnr': psnr_val,
            'ssim': ssim_val
        })

        if (i + 1) % 10 == 0:
            print(f"  Procesadas: {i + 1} imágenes...")

    # Calcular promedios
    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)
    std_psnr = np.std(psnr_values)
    std_ssim = np.std(ssim_values)

    print(f"\n{'='*60}")
    print(f"Métricas del modelo:")
    print(f"{'='*60}")
    print(f"PSNR: {mean_psnr:.4f} ± {std_psnr:.4f}")
    print(f"SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")
    print(f"{'='*60}\n")

    # Guardar métricas individuales si se especificó directorio
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

        # Guardar métricas individuales
        individual_file = os.path.join(metrics_dir, 'individual_metrics.txt')
        with open(individual_file, 'w') as f:
            f.write(f"{'Imagen':<50} {'PSNR':>10} {'SSIM':>10}\n")
            f.write(f"{'-'*72}\n")
            for metric in individual_metrics:
                f.write(f"{metric['image']:<50} {metric['psnr']:>10.4f} {metric['ssim']:>10.4f}\n")
            f.write(f"{'-'*72}\n")
            f.write(f"{'Promedio':<50} {mean_psnr:>10.4f} {mean_ssim:>10.4f}\n")
            f.write(f"{'Desviación estándar':<50} {std_psnr:>10.4f} {std_ssim:>10.4f}\n")

        print(f"✓ Métricas individuales guardadas: {individual_file}")

    return {
        'mean_psnr': mean_psnr,
        'mean_ssim': mean_ssim,
        'std_psnr': std_psnr,
        'std_ssim': std_ssim,
        'psnr_values': psnr_values,
        'ssim_values': ssim_values,
        'individual_metrics': individual_metrics
    }