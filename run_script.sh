#!/bin/bash

# Wrapper para ejecutar scripts desde la raíz del proyecto
# Uso: ./run_script.sh [nombre_script] [argumentos...]

SCRIPT_NAME="$1"
shift  # Remover el primer argumento

if [ -z "$SCRIPT_NAME" ]; then
    echo "Uso: ./run_script.sh [nombre_script] [argumentos...]"
    echo ""
    echo "Scripts disponibles:"
    echo "  train_model         - Entrenar un modelo individual"
    echo "  predict_model       - Realizar predicciones"
    echo "  train_all_models    - Entrenar todos los modelos"
    echo "  predict_all_models  - Predecir con todos los modelos"
    echo "  quick_test          - Prueba rápida"
    echo ""
    echo "Ejemplos:"
    echo "  ./run_script.sh train_model canny"
    echo "  ./run_script.sh predict_model canny_model canny"
    echo "  ./run_script.sh quick_test canny"
    exit 1
fi

# Agregar extensión .sh si no la tiene
if [[ ! "$SCRIPT_NAME" =~ \.sh$ ]]; then
    SCRIPT_NAME="${SCRIPT_NAME}.sh"
fi

SCRIPT_PATH="scripts/$SCRIPT_NAME"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: No se encontró el script: $SCRIPT_PATH"
    echo "Scripts disponibles en scripts/:"
    ls -1 scripts/*.sh | sed 's|scripts/||'
    exit 1
fi

# Ejecutar el script con los argumentos restantes
bash "$SCRIPT_PATH" "$@"
