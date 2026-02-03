#!/bin/bash

# Script para entrenar modelo Pix2Pix
# Uso: ./train_model.sh [nombre_algoritmo] [nombre_modelo] [steps]

# ============================================================================
# CONFIGURACIÓN - Modifica estos parámetros según tus necesidades
# ============================================================================

# Algoritmo a usar (canny, laplaciano, canny_inverso, etc.)
ALGORITMO="${1:-canny}"

# Nombre del modelo (si no se especifica, usa el nombre del algoritmo)
NOMBRE_MODELO="${2:-${ALGORITMO}_model}"

# Número de pasos de entrenamiento
STEPS="${3:-500000}"

# Ruta al dataset de entrada (algoritmo)
INPUT_PATH="dataset/${ALGORITMO}"

# Ruta al ground truth
GROUND_TRUTH_PATH="dataset/1051 Redimensionadas"

# Dimensiones de las imágenes
IMG_WIDTH=1024
IMG_HEIGHT=413

# Parámetros de entrenamiento
BATCH_SIZE=1
LEARNING_RATE=0.0002
LAMBDA_L1=100

# Intervalos de evaluación y guardado
EVAL_INTERVAL=1000
SAVE_INTERVAL=5000

# Directorios de checkpoints y logs
CHECKPOINT_DIR="./training_checkpoints"
LOG_DIR="./logs"

# Opciones adicionales (descomenta las que quieras usar)
# SHOW_SAMPLES="--show-samples"              # Mostrar muestras del dataset
# SHOW_MODEL_SUMMARY="--show-model-summary"  # Mostrar resumen del modelo
# RESTORE_CHECKPOINT="--restore-checkpoint"  # Restaurar desde checkpoint
# NO_AUGMENTATION="--no-augmentation"        # Desactivar data augmentation

# ============================================================================
# NO MODIFICAR A PARTIR DE AQUÍ (a menos que sepas lo que haces)
# ============================================================================

echo "============================================================"
echo "Entrenamiento Pix2Pix"
echo "============================================================"
echo "Algoritmo:       ${ALGORITMO}"
echo "Modelo:          ${NOMBRE_MODELO}"
echo "Steps:           ${STEPS}"
echo "Input path:      ${INPUT_PATH}"
echo "Ground truth:    ${GROUND_TRUTH_PATH}"
echo "============================================================"
echo ""

# Verificar que exista el directorio de entrada
if [ ! -d "$INPUT_PATH/train" ]; then
    echo "ERROR: No se encontró el directorio ${INPUT_PATH}/train"
    echo "Por favor, ejecuta primero: python organize.py"
    exit 1
fi

# Verificar que exista el directorio de ground truth
if [ ! -d "$GROUND_TRUTH_PATH" ]; then
    echo "ERROR: No se encontró el directorio ${GROUND_TRUTH_PATH}"
    exit 1
fi

# Ejecutar entrenamiento
python train.py \
    --input-path "${INPUT_PATH}" \
    --ground-truth-path "${GROUND_TRUTH_PATH}" \
    --dataset-name "${NOMBRE_MODELO}" \
    --steps ${STEPS} \
    --img-width ${IMG_WIDTH} \
    --img-height ${IMG_HEIGHT} \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --lambda-l1 ${LAMBDA_L1} \
    --eval-interval ${EVAL_INTERVAL} \
    --save-interval ${SAVE_INTERVAL} \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --log-dir "${LOG_DIR}" \
    ${SHOW_SAMPLES} \
    ${SHOW_MODEL_SUMMARY} \
    ${RESTORE_CHECKPOINT} \
    ${NO_AUGMENTATION}

echo ""
echo "============================================================"
echo "Entrenamiento completado"
echo "Resultados guardados en: resultados/${NOMBRE_MODELO}/"
echo "============================================================"
