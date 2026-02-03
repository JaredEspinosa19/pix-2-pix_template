"""
Organize Dataset Script
Organiza las imágenes en carpetas train/test según los archivos de images_list.
"""

import os
import shutil
from pathlib import Path


def read_image_list(file_path):
    """
    Lee un archivo de lista de imágenes y extrae los números.

    Args:
        file_path: Ruta al archivo (train_images.txt o test_images.txt)

    Returns:
        Lista de números de imagen (ej: ['0004', '0006', ...])
    """
    image_numbers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Solo agregar líneas no vacías
                image_numbers.append(line)
    return image_numbers


def organize_algorithm_folder(algorithm_path, train_numbers, test_numbers):
    """
    Organiza una carpeta de algoritmo en train/test.

    Args:
        algorithm_path: Ruta a la carpeta del algoritmo (ej: 'dataset/canny')
        train_numbers: Lista de números para train
        test_numbers: Lista de números para test
    """
    if not os.path.exists(algorithm_path):
        print(f"  WARNING: Carpeta no existe: {algorithm_path}")
        return

    # Crear carpetas train y test si no existen
    train_dir = os.path.join(algorithm_path, 'train')
    test_dir = os.path.join(algorithm_path, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Obtener todas las imágenes en la carpeta principal
    all_files = [f for f in os.listdir(algorithm_path)
                 if f.endswith('.png') and os.path.isfile(os.path.join(algorithm_path, f))]

    if not all_files:
        print(f"  WARNING: No se encontraron imagenes PNG en: {algorithm_path}")
        return

    train_count = 0
    test_count = 0
    skipped = 0

    for filename in all_files:
        # Extraer los últimos 4 caracteres antes de .png
        # Ej: "SEM Imaging_ETD-SE_q1_x01_y01_s0004.png" -> "0004"
        if filename.endswith('.png'):
            num_part = filename[-8:-4]  # Últimos 4 caracteres antes de .png
        else:
            print(f"    WARNING: No se pudo extraer numero de: {filename}")
            skipped += 1
            continue

        source_path = os.path.join(algorithm_path, filename)

        # Determinar si va en train o test
        if num_part in train_numbers:
            dest_path = os.path.join(train_dir, filename)
            shutil.copy2(source_path, dest_path)
            train_count += 1
        elif num_part in test_numbers:
            dest_path = os.path.join(test_dir, filename)
            shutil.copy2(source_path, dest_path)
            test_count += 1
        else:
            # Imagen no está en ninguna lista
            skipped += 1

    print(f"  OK {algorithm_path}")
    print(f"    - Train: {train_count} imagenes")
    print(f"    - Test: {test_count} imagenes")
    if skipped > 0:
        print(f"    - Omitidas: {skipped} imagenes")


def main():
    """Función principal para organizar todos los datasets."""
    print("="*60)
    print("Organizando datasets en train/test")
    print("="*60)

    # Leer listas de train y test
    base_path = 'dataset'
    train_file = os.path.join(base_path, 'images_list', 'train_images.txt')
    test_file = os.path.join(base_path, 'images_list', 'test_images.txt')

    if not os.path.exists(train_file):
        print(f"Error: No se encontró {train_file}")
        return

    if not os.path.exists(test_file):
        print(f"Error: No se encontró {test_file}")
        return

    print("\n1. Leyendo listas de imágenes...")
    train_numbers = read_image_list(train_file)
    test_numbers = read_image_list(test_file)

    print(f"  - Train: {len(train_numbers)} imagenes")
    print(f"  - Test: {len(test_numbers)} imagenes")

    # Lista de carpetas de algoritmos a organizar
    algorithm_folders = [
        'canny',
        'canny_inverso',
        'canny_inverso_erosion',
        'laplaciano',
        'laplaciano_inverso',
        'laplaciano_inverso_erosion'
    ]

    print("\n2. Organizando carpetas de algoritmos...")
    for folder in algorithm_folders:
        algorithm_path = os.path.join(base_path, folder)
        organize_algorithm_folder(algorithm_path, train_numbers, test_numbers)

    print("\n" + "="*60)
    print("Organización completada")
    print("="*60)

    # Verificar que no se hayan movido las imágenes de ground truth
    gt_path = os.path.join(base_path, '1051 Redimensionadas')
    if os.path.exists(gt_path):
        gt_files = [f for f in os.listdir(gt_path) if f.endswith('.png')]
        print(f"\nOK Ground Truth intacto: {len(gt_files)} imagenes en '{gt_path}'")
    else:
        print(f"\nWARNING: Advertencia: No se encontro carpeta de Ground Truth: {gt_path}")


if __name__ == '__main__':
    main()