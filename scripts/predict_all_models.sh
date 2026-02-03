#!/bin/bash

# Script para realizar predicciones con todos los modelos entrenados
# Uso: ./predict_all_models.sh

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Ground truth path
GROUND_TRUTH_PATH="dataset/1051 Redimensionadas"

# Lista de modelos a evaluar (algoritmo:nombre_modelo)
MODELOS=(
    "canny:canny_model"
    "canny_inverso:canny_inverso_model"
    "canny_inverso_erosion:canny_inv_erosion_model"
    "laplaciano:laplaciano_model"
    "laplaciano_inverso_erosion:laplaciano_inv_erosion_model"
)

# ============================================================================
# NO MODIFICAR A PARTIR DE AQUÍ
# ============================================================================

echo "============================================================"
echo "Predicción de Todos los Modelos"
echo "============================================================"
echo "Modelos a evaluar: ${#MODELOS[@]}"
echo ""
for modelo in "${MODELOS[@]}"; do
    IFS=':' read -r algo nombre <<< "$modelo"
    echo "  - ${nombre} (${algo})"
done
echo "============================================================"
echo ""

read -p "¿Desea continuar? (s/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[SsYy]$ ]]; then
    echo "Operación cancelada"
    exit 1
fi

# Contador de modelos
TOTAL=${#MODELOS[@]}
CURRENT=0
SUCCESS=0
FAILED=0

# Evaluar cada modelo
for MODELO_INFO in "${MODELOS[@]}"; do
    IFS=':' read -r ALGORITMO NOMBRE_MODELO <<< "$MODELO_INFO"
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "============================================================"
    echo "Evaluando modelo ${CURRENT}/${TOTAL}: ${NOMBRE_MODELO}"
    echo "============================================================"
    echo ""

    WEIGHTS_PATH="resultados/${NOMBRE_MODELO}/weights/best_generator_weights.h5"

    # Verificar que existan los pesos
    if [ ! -f "$WEIGHTS_PATH" ]; then
        echo "WARNING: No se encontró ${WEIGHTS_PATH} - SALTANDO"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Verificar que exista el dataset de test
    if [ ! -d "dataset/${ALGORITMO}/test" ]; then
        echo "WARNING: No se encontró dataset/${ALGORITMO}/test - SALTANDO"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Realizar predicción
    python predict.py \
        --weights "${WEIGHTS_PATH}" \
        --test-dataset "dataset/${ALGORITMO}" \
        --ground-truth-path "${GROUND_TRUTH_PATH}" \
        --output-dir "resultados/${NOMBRE_MODELO}" \
        --img-width 1024 \
        --img-height 413

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Modelo ${NOMBRE_MODELO} evaluado"
        SUCCESS=$((SUCCESS + 1))
    else
        echo ""
        echo "✗ Error al evaluar ${NOMBRE_MODELO}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "EVALUACIÓN COMPLETADA"
echo "============================================================"
echo "Total:        ${TOTAL}"
echo "Exitosos:     ${SUCCESS}"
echo "Fallidos:     ${FAILED}"
echo "============================================================"

# Mostrar resumen de métricas
echo ""
echo "============================================================"
echo "RESUMEN DE MÉTRICAS"
echo "============================================================"
echo ""

for MODELO_INFO in "${MODELOS[@]}"; do
    IFS=':' read -r ALGORITMO NOMBRE_MODELO <<< "$MODELO_INFO"
    METRICS_FILE="resultados/${NOMBRE_MODELO}/metricas/test_metrics_summary.txt"

    if [ -f "$METRICS_FILE" ]; then
        echo "--- ${NOMBRE_MODELO} ---"
        tail -n 2 "$METRICS_FILE"
        echo ""
    fi
done

echo "============================================================"
