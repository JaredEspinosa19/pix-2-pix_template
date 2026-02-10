#!/bin/bash

# Script para entrenar todos los modelos secuencialmente
# Uso: ./train_all_models.sh [steps]
# Nota: Este script debe ejecutarse desde la raíz del proyecto

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Cambiar al directorio raíz del proyecto (un nivel arriba de scripts/)
cd "$(dirname "$0")/.." || exit 1

# Número de pasos para cada modelo (default: 500000)
STEPS="${1:-500000}"

# Ground truth path
GROUND_TRUTH_PATH="dataset/1051 Redimensionadas"

# Lista de algoritmos a entrenar
ALGORITMOS=(
    "canny"
    "canny_inverso"
    "canny_inverso_erosion"
    "laplaciano"
    "laplaciano_inverso_erosion"
)

# ============================================================================
# NO MODIFICAR A PARTIR DE AQUÍ
# ============================================================================

echo "============================================================"
echo "Entrenamiento de Todos los Modelos"
echo "============================================================"
echo "Steps por modelo: ${STEPS}"
echo "Modelos a entrenar: ${#ALGORITMOS[@]}"
echo ""
for algo in "${ALGORITMOS[@]}"; do
    echo "  - ${algo}"
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
TOTAL=${#ALGORITMOS[@]}
CURRENT=0

# Entrenar cada modelo
for ALGORITMO in "${ALGORITMOS[@]}"; do
    CURRENT=$((CURRENT + 1))
    NOMBRE_MODELO="${ALGORITMO}_model"

    echo ""
    echo "============================================================"
    echo "Entrenando modelo ${CURRENT}/${TOTAL}: ${NOMBRE_MODELO}"
    echo "============================================================"
    echo ""

    # Verificar que exista el directorio
    if [ ! -d "dataset/${ALGORITMO}/train" ]; then
        echo "WARNING: No se encontró dataset/${ALGORITMO}/train - SALTANDO"
        continue
    fi

    # Entrenar modelo
    python train.py \
        --input-path "dataset/${ALGORITMO}" \
        --ground-truth-path "${GROUND_TRUTH_PATH}" \
        --dataset-name "${NOMBRE_MODELO}" \
        --steps ${STEPS} \
        --img-width 1024 \
        --img-height 413 \
        --eval-interval 1000 \
        --save-interval 5000

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Modelo ${NOMBRE_MODELO} completado"
    else
        echo ""
        echo "✗ Error al entrenar ${NOMBRE_MODELO}"
        read -p "¿Desea continuar con el siguiente modelo? (s/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[SsYy]$ ]]; then
            exit 1
        fi
    fi
done

echo ""
echo "============================================================"
echo "TODOS LOS MODELOS COMPLETADOS"
echo "============================================================"
echo "Modelos entrenados: ${CURRENT}/${TOTAL}"
echo "Resultados en: resultados/"
