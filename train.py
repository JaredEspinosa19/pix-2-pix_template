#!/usr/bin/env python
"""
Script de entrada para entrenamiento.
Ejecuta src/main.py
"""
import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Importar y ejecutar main
from main import main

if __name__ == '__main__':
    main()
