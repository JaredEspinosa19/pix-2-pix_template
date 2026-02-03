#!/bin/bash

# Script para hacer una prueba rápida del entrenamiento
# Entrena por pocos pasos para verificar que todo funciona
# Uso: ./quick_test.sh [algoritmo]

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

ALGORITMO="${1:-canny}"
NOMBRE_MODELO="${ALGORITMO}_test"
STEPS=100  # Solo 100 pasos para prueba rápida
GROUND_TRUTH_PATH="dataset/1051 Redimensionadas"
INPUT_PATH="dataset/${ALGORITMO}"

# ============================================================================
# NO MODIFICAR A PARTIR DE AQUÍ
# ============================================================================

echo "============================================================"
echo "Prueba Rápida de Entrenamiento"
echo "============================================================"
echo "Algoritmo:    ${ALGORITMO}"
echo "Steps:        ${STEPS} (solo para prueba)"
echo "Modelo:       ${NOMBRE_MODELO}"
echo "============================================================"
echo ""

# Verificar directorios
if [ ! -d "$INPUT_PATH/train" ]; then
    echo "ERROR: No se encontró ${INPUT_PATH}/train"
    exit 1
fi

# Ejecutar con visualización de muestras
python train.py \
    --input-path "${INPUT_PATH}" \
    --ground-truth-path "${GROUND_TRUTH_PATH}" \
    --dataset-name "${NOMBRE_MODELO}" \
    --steps ${STEPS} \
    --eval-interval 50 \
    --save-interval 50 \
    --show-samples \
    --show-model-summary

echo ""
echo "============================================================"
echo "Prueba completada"
echo "Si no hubo errores, puedes proceder con entrenamiento completo"
echo "============================================================"
echo ""
echo "Para entrenar el modelo completo, ejecuta:"
echo "  ./train_model.sh ${ALGORITMO}"
