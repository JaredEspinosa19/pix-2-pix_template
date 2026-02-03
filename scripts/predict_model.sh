#!/bin/bash

# Script para realizar predicciones con modelo Pix2Pix entrenado
# Uso: ./predict_model.sh [nombre_modelo] [algoritmo]

# ============================================================================
# CONFIGURACIÓN - Modifica estos parámetros según tus necesidades
# ============================================================================

# Nombre del modelo entrenado
NOMBRE_MODELO="${1:-canny_model}"

# Algoritmo usado (para encontrar el test dataset)
ALGORITMO="${2:-canny}"

# Ruta a los pesos del modelo
WEIGHTS_PATH="resultados/${NOMBRE_MODELO}/weights/best_generator_weights.h5"

# Ruta al dataset de prueba
TEST_DATASET="dataset/${ALGORITMO}"

# Ruta al ground truth
GROUND_TRUTH_PATH="dataset/1051 Redimensionadas"

# Directorio de salida (usa el mismo del modelo)
OUTPUT_DIR="resultados/${NOMBRE_MODELO}"

# Dimensiones de las imágenes
IMG_WIDTH=1024
IMG_HEIGHT=413

# Opciones adicionales (descomenta la que quieras usar)
# VISUALIZE="--visualize"  # Visualizar predicciones

# ============================================================================
# NO MODIFICAR A PARTIR DE AQUÍ (a menos que sepas lo que haces)
# ============================================================================

echo "============================================================"
echo "Predicción Pix2Pix"
echo "============================================================"
echo "Modelo:          ${NOMBRE_MODELO}"
echo "Algoritmo:       ${ALGORITMO}"
echo "Weights:         ${WEIGHTS_PATH}"
echo "Test dataset:    ${TEST_DATASET}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# Verificar que existan los pesos
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "ERROR: No se encontró el archivo de pesos: ${WEIGHTS_PATH}"
    echo "Por favor, verifica que el modelo esté entrenado"
    exit 1
fi

# Verificar que exista el directorio de test
if [ ! -d "$TEST_DATASET/test" ]; then
    echo "ERROR: No se encontró el directorio ${TEST_DATASET}/test"
    exit 1
fi

# Ejecutar predicción
python predict.py \
    --weights "${WEIGHTS_PATH}" \
    --test-dataset "${TEST_DATASET}" \
    --ground-truth-path "${GROUND_TRUTH_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --img-width ${IMG_WIDTH} \
    --img-height ${IMG_HEIGHT} \
    ${VISUALIZE}

echo ""
echo "============================================================"
echo "Predicción completada"
echo "Imágenes guardadas en: ${OUTPUT_DIR}/imagenes_prueba/"
echo "Métricas guardadas en: ${OUTPUT_DIR}/metricas/"
echo "============================================================"
