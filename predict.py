#!/usr/bin/env python
"""
Script de entrada para inferencia.
Ejecuta src/inference.py
"""
import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Importar y ejecutar main
from inference import main

if __name__ == '__main__':
    main()
