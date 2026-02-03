"""
Main Script
Script principal para entrenar el modelo Pix2Pix.
"""

import tensorflow as tf
import argparse
import os
from dataset_loader import validate_and_load_dataset
from network import create_pix2pix_model, create_optimizers
from training import Pix2PixTrainer, calculate_metrics


def check_gpu():
    """Verifica y muestra información sobre GPUs disponibles."""
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n{'='*60}")
    print(f"Configuración de Hardware")
    print(f"{'='*60}")
    print(f"GPUs disponibles: {len(gpus)}")

    if gpus:
        print("TensorFlow utilizará GPU")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print("TensorFlow utilizará CPU")

    print(f"{'='*60}\n")


def train(args):
    """
    Función principal de entrenamiento.

    Args:
        args: Argumentos de línea de comandos
    """
    # Verificar GPU
    check_gpu()

    # Paso 1: Validar y cargar dataset
    print(f"\n{'='*60}")
    print("PASO 1: Validación y Carga de Dataset")
    print(f"{'='*60}")

    train_dataset, test_dataset, loader = validate_and_load_dataset(
        input_path=args.input_path,
        ground_truth_path=args.ground_truth_path,
        img_width=args.img_width,
        img_height=args.img_height,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        show_samples=args.show_samples,
        use_augmentation=args.use_augmentation
    )

    # Paso 2: Crear modelo
    print(f"\n{'='*60}")
    print("PASO 2: Creación del Modelo")
    print(f"{'='*60}")

    generator, discriminator, loss_fn = create_pix2pix_model(
        img_height=args.img_height,
        img_width=args.img_width,
        output_channels=1,
        lambda_l1=args.lambda_l1
    )

    # Crear optimizadores
    generator_optimizer, discriminator_optimizer = create_optimizers(
        learning_rate=args.learning_rate,
        beta_1=args.beta_1
    )

    # Mostrar resúmenes de los modelos si se solicita
    if args.show_model_summary:
        print("\n" + "="*60)
        print("Resumen del Generador:")
        print("="*60)
        generator.summary()

        print("\n" + "="*60)
        print("Resumen del Discriminador:")
        print("="*60)
        discriminator.summary()

    # Paso 3: Configurar entrenamiento
    print(f"\n{'='*60}")
    print("PASO 3: Configuración del Entrenamiento")
    print(f"{'='*60}")

    trainer = Pix2PixTrainer(
        generator=generator,
        discriminator=discriminator,
        loss_fn=loss_fn,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        dataset_name=args.dataset_name,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

    # Restaurar checkpoint si existe y se solicita
    if args.restore_checkpoint:
        trainer.restore_checkpoint()

    # Paso 4: Entrenar
    print(f"\n{'='*60}")
    print("PASO 4: Entrenamiento")
    print(f"{'='*60}")
    print(f"Configuración:")
    print(f"  - Pasos totales: {args.steps}")
    print(f"  - Intervalo de evaluación: {args.eval_interval}")
    print(f"  - Intervalo de guardado: {args.save_interval}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")

    trainer.fit(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        steps=args.steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval
    )

    # Paso 5: Evaluación final
    print(f"\n{'='*60}")
    print("PASO 5: Evaluación Final")
    print(f"{'='*60}")

    metrics = calculate_metrics(generator, test_dataset)

    # Guardar modelo final
    final_model_path = f"{args.dataset_name}_generator_final.keras"
    generator.save(final_model_path)
    print(f"\n✓ Modelo final guardado: {final_model_path}")

    final_disc_path = f"{args.dataset_name}_discriminator_final.keras"
    discriminator.save(final_disc_path)
    print(f"✓ Discriminador final guardado: {final_disc_path}")

    print(f"\n{'='*60}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*60}\n")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Entrenar modelo Pix2Pix para traducción imagen a imagen'
    )

    # Argumentos del dataset
    parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='Ruta a la carpeta con imágenes de entrada (ej: dataset/canny)'
    )
    parser.add_argument(
        '--ground-truth-path',
        type=str,
        default='dataset/1051 Redimensionadas',
        help='Ruta a las imágenes ground truth (default: dataset/1051 Redimensionadas)'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default=None,
        help='Nombre del dataset (para guardar resultados). Si no se especifica, se usa el nombre del directorio de entrada.'
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

    # Argumentos de entrenamiento
    parser.add_argument(
        '--steps',
        type=int,
        default=500000,
        help='Número total de pasos de entrenamiento'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Tamaño del batch'
    )
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=400,
        help='Tamaño del buffer para shuffle'
    )
    parser.add_argument(
        '--no-augmentation',
        action='store_true',
        help='Desactivar data augmentation (random crop y flip). Por defecto está activado.'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-4,
        help='Tasa de aprendizaje'
    )
    parser.add_argument(
        '--beta-1',
        type=float,
        default=0.5,
        help='Parámetro beta_1 del optimizador Adam'
    )
    parser.add_argument(
        '--lambda-l1',
        type=int,
        default=100,
        help='Peso de la pérdida L1'
    )

    # Argumentos de evaluación y guardado
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1000,
        help='Intervalo de pasos para evaluar el modelo'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=5000,
        help='Intervalo de pasos para guardar checkpoints'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./training_checkpoints',
        help='Directorio para guardar checkpoints'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Directorio para logs de TensorBoard'
    )

    # Argumentos opcionales
    parser.add_argument(
        '--restore-checkpoint',
        action='store_true',
        help='Restaurar desde el último checkpoint'
    )
    parser.add_argument(
        '--show-model-summary',
        action='store_true',
        help='Mostrar resumen de los modelos'
    )
    parser.add_argument(
        '--show-samples',
        action='store_true',
        help='Mostrar imágenes de muestra del dataset antes de entrenar'
    )

    args = parser.parse_args()

    # Si no se especifica dataset_name, usar el nombre del directorio de entrada
    if args.dataset_name is None:
        args.dataset_name = os.path.basename(args.input_path.rstrip('/\\'))

    # Convertir --no-augmentation a use_augmentation (invertir el flag)
    args.use_augmentation = not args.no_augmentation

    # Ejecutar entrenamiento
    train(args)


if __name__ == '__main__':
    main()