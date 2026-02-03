#!/usr/bin/env python
"""
Script de entrada para organización de datasets.
Ejecuta src/organize_dataset.py
"""
import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Importar y ejecutar main
from organize_dataset import main

if __name__ == '__main__':
    # Preguntar confirmación antes de ejecutar
    print("\n" + "="*60)
    print("Este script organizará las imágenes en carpetas train/test")
    print("Las imágenes se COPIARÁN (no se moverán) a las carpetas correspondientes")
    print("="*60)

    response = input("\n¿Desea continuar? (s/n): ")
    if response.lower() in ['s', 'si', 'sí', 'y', 'yes']:
        main()
    else:
        print("Operación cancelada.")
