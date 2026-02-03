"""
Inference Script
Script para realizar predicciones con modelos Pix2Pix entrenados.
"""

import tensorflow as tf
import argparse
import os
from dataset_loader import DatasetLoader
from network import Generator
from training import Pix2PixInference, calculate_metrics


def main():
    """Función principal de inferencia."""
    parser = argparse.ArgumentParser(
        description='Realizar predicciones con modelo Pix2Pix entrenado'
    )

    # Argumentos del modelo
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Ruta al archivo de pesos del generador (.h5 o .keras)'
    )
    parser.add_argument(
        '--img-width',
        type=int,
        default=1024,
        help='Ancho de las imágenes'
    )
    parser.add_argument(
        '--img-height',
        type=int,
        default=413,
        help='Alto de las imágenes'
    )

    # Argumentos de entrada
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Directorio con imágenes para predecir'
    )
    parser.add_argument(
        '--input-image',
        type=str,
        help='Imagen individual para predecir'
    )
    parser.add_argument(
        '--test-dataset',
        type=str,
        help='Ruta al dataset de prueba (carpeta con train/test, ej: dataset/canny)'
    )
    parser.add_argument(
        '--ground-truth-path',
        type=str,
        default='dataset/1051 Redimensionadas',
        help='Ruta a las imágenes ground truth (default: dataset/1051 Redimensionadas)'
    )

    # Argumentos de salida
    parser.add_argument(
        '--output-dir',
        type=str,
        default='predictions',
        help='Directorio donde guardar las predicciones'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualizar predicciones'
    )

    args = parser.parse_args()

    # Verificar que se proporcione al menos una entrada
    if not any([args.input_dir, args.input_image, args.test_dataset]):
        parser.error('Debe proporcionar --input-dir, --input-image o --test-dataset')

    # Crear generador
    print(f"\n{'='*60}")
    print("Cargando modelo...")
    print(f"{'='*60}")

    generator = Generator(
        output_channels=1,
        img_height=args.img_height,
        img_width=args.img_width
    )

    # Crear dataset loader (se necesita para las funciones de carga)
    # Para inferencia, usamos paths dummy ya que cargaremos imágenes individualmente
    loader = DatasetLoader(
        input_path='.',
        ground_truth_path='.',
        img_width=args.img_width,
        img_height=args.img_height
    )

    # Crear objeto de inferencia
    inference = Pix2PixInference(generator, loader)

    # Cargar pesos
    inference.load_weights(args.weights)

    # Modo 1: Predecir imagen individual
    if args.input_image:
        print(f"\n{'='*60}")
        print("Prediciendo imagen individual...")
        print(f"{'='*60}")

        # Crear directorio de salida
        os.makedirs(args.output_dir, exist_ok=True)

        output_path = os.path.join(
            args.output_dir,
            'pred_' + os.path.basename(args.input_image)
        )

        if args.visualize:
            inference.visualize_prediction(
                args.input_image,
                save_path=output_path.replace('.png', '_vis.png')
            )
        else:
            input_image, prediction = inference.predict_image(args.input_image)
            inference._save_prediction(prediction, output_path)
            print(f"✓ Predicción guardada: {output_path}")

    # Modo 2: Predecir directorio completo
    if args.input_dir:
        print(f"\n{'='*60}")
        print("Prediciendo directorio completo...")
        print(f"{'='*60}")

        inference.predict_dataset(args.input_dir, args.output_dir)

    # Modo 3: Evaluar en dataset de prueba
    if args.test_dataset:
        print(f"\n{'='*60}")
        print("Evaluando en dataset de prueba...")
        print(f"{'='*60}")

        # Determinar paths
        input_test_path = args.test_dataset
        gt_path = args.ground_truth_path if hasattr(args, 'ground_truth_path') and args.ground_truth_path else 'dataset/1051 Redimensionadas'

        # Obtener nombres de archivos del dataset de prueba
        test_images_dir = os.path.join(args.test_dataset, 'test')
        image_files = []
        if os.path.exists(test_images_dir):
            image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.png')])

        # Cargar dataset de prueba
        test_loader = DatasetLoader(
            input_path=input_test_path,
            ground_truth_path=gt_path,
            img_width=args.img_width,
            img_height=args.img_height,
            batch_size=1
        )
        test_dataset = test_loader.create_test_dataset()

        # Configurar directorio de métricas
        metrics_dir = os.path.join(args.output_dir, 'metricas')
        os.makedirs(metrics_dir, exist_ok=True)

        # Calcular métricas con nombres de archivo y guardar individuales
        metrics = calculate_metrics(generator, test_dataset, image_names=image_files, metrics_dir=metrics_dir)

        # Guardar resumen de métricas
        metrics_file = os.path.join(metrics_dir, 'test_metrics_summary.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Resumen de Métricas del Dataset de Prueba\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Total de imágenes: {len(image_files)}\n\n")
            f.write(f"PSNR: {metrics['mean_psnr']:.4f} ± {metrics['std_psnr']:.4f}\n")
            f.write(f"SSIM: {metrics['mean_ssim']:.4f} ± {metrics['std_ssim']:.4f}\n")

        print(f"✓ Resumen de métricas guardado: {metrics_file}")

        # Generar predicciones del dataset de prueba
        if args.output_dir and os.path.exists(test_images_dir):
            print("\nGenerando predicciones del dataset de prueba...")
            predictions_dir = os.path.join(args.output_dir, 'imagenes_prueba')
            inference.predict_dataset(test_images_dir, predictions_dir)

    print(f"\n{'='*60}")
    print("INFERENCIA COMPLETADA")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()