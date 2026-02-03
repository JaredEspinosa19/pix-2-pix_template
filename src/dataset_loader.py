"""
Dataset Loader Module
Valida la estructura del dataset y proporciona funciones para cargar y preprocesar imágenes.
"""

import tensorflow as tf
import os
from pathlib import Path


class DatasetValidator:
    """Valida la estructura y contenido del dataset."""

    def __init__(self, input_path, ground_truth_path):
        """
        Args:
            input_path: Ruta a la carpeta con imágenes de entrada (ej: 'dataset/canny')
            ground_truth_path: Ruta a las imágenes ground truth ('dataset/1051 Redimensionadas')
        """
        self.input_path = input_path
        self.ground_truth_path = ground_truth_path

    def validate_structure(self):
        """
        Valida que el dataset tenga la estructura correcta:
        - Carpetas train/ y test/ en input_path
        - Carpeta ground_truth_path
        - Archivos .png en ambas carpetas
        """
        print(f"Validando estructura del dataset...")
        print(f"  Input: {self.input_path}")
        print(f"  Ground Truth: {self.ground_truth_path}")

        # Verificar que existan los directorios
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"El directorio de entrada no existe: {self.input_path}")

        if not os.path.exists(self.ground_truth_path):
            raise FileNotFoundError(f"El directorio de ground truth no existe: {self.ground_truth_path}")

        # Verificar carpetas train y test en input_path
        train_path = os.path.join(self.input_path, 'train')
        test_path = os.path.join(self.input_path, 'test')

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"No se encontró la carpeta 'train' en: {self.input_path}")

        if not os.path.exists(test_path):
            raise FileNotFoundError(f"No se encontró la carpeta 'test' en: {self.input_path}")

        # Contar archivos
        train_files = [f for f in os.listdir(train_path) if f.endswith('.png')]
        test_files = [f for f in os.listdir(test_path) if f.endswith('.png')]
        gt_files = [f for f in os.listdir(self.ground_truth_path) if f.endswith('.png')]

        if len(train_files) == 0:
            raise ValueError(f"No se encontraron imágenes .png en: {train_path}")
        if len(test_files) == 0:
            raise ValueError(f"No se encontraron imágenes .png en: {test_path}")
        if len(gt_files) == 0:
            raise ValueError(f"No se encontraron imágenes ground truth en: {self.ground_truth_path}")

        print(f"✓ Estructura correcta")
        print(f"  - Train: {len(train_files)} imágenes")
        print(f"  - Test: {len(test_files)} imágenes")
        print(f"  - Ground Truth: {len(gt_files)} imágenes")

        # Verificar dimensiones
        sample_input = os.path.join(train_path, train_files[0])
        sample_gt = os.path.join(self.ground_truth_path, train_files[0])

        if os.path.exists(sample_gt):
            self._validate_image_format(sample_input, sample_gt)
        else:
            print(f"  ⚠ Advertencia: No se encontró imagen ground truth correspondiente")

        return True

    def _validate_image_format(self, input_path, gt_path):
        """Valida que las imágenes tengan el formato esperado."""
        input_img = tf.io.read_file(input_path)
        input_img = tf.io.decode_png(input_img, channels=1)

        gt_img = tf.io.read_file(gt_path)
        gt_img = tf.io.decode_png(gt_img, channels=1)

        print(f"  - Input: {input_img.shape[1]}x{input_img.shape[0]}, canales: {input_img.shape[2]}")
        print(f"  - Ground Truth: {gt_img.shape[1]}x{gt_img.shape[0]}, canales: {gt_img.shape[2]}")

        return True


class DatasetLoader:
    """Carga y preprocesa el dataset para entrenamiento."""

    def __init__(self, input_path, ground_truth_path, img_width=1024, img_height=413,
                 buffer_size=400, batch_size=1, use_augmentation=True):
        """
        Args:
            input_path: Ruta a la carpeta con imágenes de entrada (ej: 'dataset/canny')
            ground_truth_path: Ruta a las imágenes ground truth ('dataset/1051 Redimensionadas')
            img_width: Ancho objetivo de las imágenes
            img_height: Alto objetivo de las imágenes
            buffer_size: Tamaño del buffer para shuffle
            batch_size: Tamaño del batch
            use_augmentation: Si es True, aplica data augmentation (jitter, flip). Si es False, solo redimensiona.
        """
        self.input_path = input_path
        self.ground_truth_path = ground_truth_path
        self.IMG_WIDTH = img_width
        self.IMG_HEIGHT = img_height
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.use_augmentation = use_augmentation

    def load(self, input_file):
        """
        Lee una imagen de entrada y su correspondiente ground truth.

        Args:
            input_file: Ruta al archivo de entrada (tensor de string)

        Returns:
            input_image: Imagen de entrada procesada
            real_image: Imagen ground truth procesada
        """
        # Obtener nombre del archivo
        filename = tf.strings.split(input_file, os.sep)[-1]

        # Construir ruta al ground truth
        gt_file = tf.strings.join([self.ground_truth_path, filename], separator=os.sep)

        # Leer imagen de entrada
        input_image = tf.io.read_file(input_file)
        input_image = tf.io.decode_png(input_image, channels=1)
        input_image = tf.cast(input_image, tf.float32)

        # Leer imagen ground truth
        real_image = tf.io.read_file(gt_file)
        real_image = tf.io.decode_png(real_image, channels=1)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image

    def resize(self, input_image, real_image, height, width):
        """Redimensiona las imágenes usando interpolación nearest neighbor."""
        input_image = tf.image.resize(
            input_image, [height, width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        real_image = tf.image.resize(
            real_image, [height, width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        return input_image, real_image

    def random_crop(self, input_image, real_image):
        """Recorta aleatoriamente las imágenes al tamaño especificado."""
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, self.IMG_HEIGHT, self.IMG_WIDTH, 1]
        )
        return cropped_image[0], cropped_image[1]

    def normalize(self, input_image, real_image):
        """Normaliza las imágenes al rango [-1, 1]."""
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1
        return input_image, real_image

    @tf.function()
    def random_jitter(self, input_image, real_image):
        """
        Aplica data augmentation:
        - Redimensiona a un tamaño mayor
        - Recorta aleatoriamente al tamaño original
        - Voltea horizontalmente con 50% de probabilidad
        """
        # Redimensionar a un tamaño mayor
        input_image, real_image = self.resize(
            input_image, real_image,
            int(self.IMG_HEIGHT * 1.5),
            int(self.IMG_WIDTH * 1.5)
        )

        # Recortar aleatoriamente
        input_image, real_image = self.random_crop(input_image, real_image)

        # Volteo aleatorio
        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    def load_image_train(self, image_file):
        """
        Pipeline completo de carga para entrenamiento.

        Aplica augmentation (jitter + flip) si use_augmentation=True,
        de lo contrario solo redimensiona a tamaño exacto.
        """
        input_image, real_image = self.load(image_file)

        if self.use_augmentation:
            # Aplicar data augmentation: random crop y random flip
            input_image, real_image = self.random_jitter(input_image, real_image)
        else:
            # Sin augmentation: solo redimensionar al tamaño exacto
            input_image, real_image = self.resize(
                input_image, real_image,
                self.IMG_HEIGHT, self.IMG_WIDTH
            )

        input_image, real_image = self.normalize(input_image, real_image)
        return input_image, real_image

    def load_image_test(self, image_file):
        """Pipeline completo de carga para prueba (sin augmentation)."""
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.resize(
            input_image, real_image,
            self.IMG_HEIGHT, self.IMG_WIDTH
        )
        input_image, real_image = self.normalize(input_image, real_image)
        return input_image, real_image

    def create_train_dataset(self):
        """Crea el dataset de entrenamiento con augmentation y batching."""
        train_path = os.path.join(self.input_path, 'train', '*.png')
        train_dataset = tf.data.Dataset.list_files(train_path, shuffle=False)
        train_dataset = train_dataset.map(
            self.load_image_train,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        train_dataset = train_dataset.shuffle(self.BUFFER_SIZE)
        train_dataset = train_dataset.batch(self.BATCH_SIZE)
        return train_dataset

    def create_test_dataset(self):
        """Crea el dataset de prueba sin augmentation."""
        test_path = os.path.join(self.input_path, 'test', '*.png')
        try:
            test_dataset = tf.data.Dataset.list_files(test_path, shuffle=False)
        except tf.errors.InvalidArgumentError:
            test_dataset = tf.data.Dataset.list_files(test_path, shuffle=False)

        test_dataset = test_dataset.map(self.load_image_test)
        test_dataset = test_dataset.batch(self.BATCH_SIZE)
        return test_dataset

    def get_datasets(self):
        """Retorna ambos datasets (train y test)."""
        print(f"\nCargando datasets...")
        print(f"  Input: {self.input_path}")
        print(f"  Ground Truth: {self.ground_truth_path}")
        train_ds = self.create_train_dataset()
        test_ds = self.create_test_dataset()
        print("✓ Datasets cargados exitosamente")
        return train_ds, test_ds


def validate_and_load_dataset(input_path, ground_truth_path, img_width=1024, img_height=413,
                               buffer_size=400, batch_size=1, show_samples=False, use_augmentation=True):
    """
    Función principal que valida y carga el dataset.

    Args:
        input_path: Ruta a la carpeta con imágenes de entrada (ej: 'dataset/canny')
        ground_truth_path: Ruta a las imágenes ground truth ('dataset/1051 Redimensionadas')
        img_width: Ancho de las imágenes
        img_height: Alto de las imágenes
        buffer_size: Tamaño del buffer para shuffle
        batch_size: Tamaño del batch
        show_samples: Si es True, muestra imágenes de muestra de train y test
        use_augmentation: Si es True, aplica data augmentation (jitter + flip) en train.
                         Si es False, solo redimensiona sin recortes ni volteos.

    Returns:
        train_dataset, test_dataset, loader: Datasets de TensorFlow listos para usar y el loader
    """
    # Paso 1: Validar estructura
    validator = DatasetValidator(input_path, ground_truth_path)
    validator.validate_structure()

    # Paso 2: Cargar datasets
    loader = DatasetLoader(input_path, ground_truth_path, img_width, img_height,
                          buffer_size, batch_size, use_augmentation)
    train_ds, test_ds = loader.get_datasets()

    # Mostrar información sobre augmentation
    if use_augmentation:
        print("  - Data Augmentation: ACTIVADO (random crop + random flip)")
    else:
        print("  - Data Augmentation: DESACTIVADO (solo resize)")

    # Paso 3: Mostrar muestras si se solicita
    if show_samples:
        visualize_dataset_samples(train_ds, test_ds)

    return train_ds, test_ds, loader


def visualize_dataset_samples(train_dataset, test_dataset):
    """
    Visualiza una muestra de train y test dataset.

    Args:
        train_dataset: Dataset de entrenamiento
        test_dataset: Dataset de prueba
    """
    try:
        from matplotlib import pyplot as plt

        print(f"\n{'='*60}")
        print("Visualizando muestras del dataset")
        print(f"{'='*60}\n")

        # Obtener una muestra de train
        train_sample = next(iter(train_dataset.take(1)))
        train_input, train_target = train_sample

        # Obtener una muestra de test
        test_sample = next(iter(test_dataset.take(1)))
        test_input, test_target = test_sample

        # Crear figura con 2 filas (train, test) y 2 columnas (input, ground truth)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Convertir de [-1, 1] a [0, 1] para visualización
        def denormalize(img):
            return (img + 1) / 2.0

        # Fila 1: Train
        axes[0, 0].imshow(denormalize(train_input[0, :, :, 0]), cmap='gray')
        axes[0, 0].set_title('Train - Input Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(denormalize(train_target[0, :, :, 0]), cmap='gray')
        axes[0, 1].set_title('Train - Ground Truth', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # Fila 2: Test
        axes[1, 0].imshow(denormalize(test_input[0, :, :, 0]), cmap='gray')
        axes[1, 0].set_title('Test - Input Image', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(denormalize(test_target[0, :, :, 0]), cmap='gray')
        axes[1, 1].set_title('Test - Ground Truth', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
        print("✓ Muestras visualizadas y guardadas en 'dataset_samples.png'")

        # Mostrar información de las imágenes
        print(f"\nInformación de las muestras:")
        print(f"  Train Input shape: {train_input.shape}")
        print(f"  Train Target shape: {train_target.shape}")
        print(f"  Train Input range: [{train_input.numpy().min():.3f}, {train_input.numpy().max():.3f}]")
        print(f"  Train Target range: [{train_target.numpy().min():.3f}, {train_target.numpy().max():.3f}]")
        print(f"\n  Test Input shape: {test_input.shape}")
        print(f"  Test Target shape: {test_target.shape}")
        print(f"  Test Input range: [{test_input.numpy().min():.3f}, {test_input.numpy().max():.3f}]")
        print(f"  Test Target range: [{test_target.numpy().min():.3f}, {test_target.numpy().max():.3f}]")
        print(f"{'='*60}\n")

        # Intentar mostrar en notebook si está disponible
        try:
            from IPython.display import display
            plt.show()
        except:
            pass  # No está en notebook, solo guardar

    except Exception as e:
        print(f"⚠ No se pudieron visualizar las muestras: {e}")
        print("Continuando con el entrenamiento...")