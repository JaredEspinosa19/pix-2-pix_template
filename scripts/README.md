# Scripts de Bash - Guía de Uso

Este proyecto incluye varios scripts de bash para facilitar el entrenamiento y la predicción de modelos.

## Scripts Disponibles

### 1. `train_model.sh` - Entrenar un Modelo Individual

Entrena un modelo específico con parámetros configurables.

**Uso básico:**
```bash
./scripts/train_model.sh [algoritmo] [nombre_modelo] [steps]
```

**Ejemplos:**
```bash
# Entrenar canny con configuración por defecto
./scripts/train_model.sh canny

# Entrenar laplaciano con nombre personalizado
./scripts/train_model.sh laplaciano laplaciano_v2

# Entrenar con 1 millón de steps
./scripts/train_model.sh canny canny_1M 1000000
```

**Parámetros configurables** (editar el archivo):
```bash
# Dimensiones de imagen
IMG_WIDTH=1024
IMG_HEIGHT=413

# Parámetros de entrenamiento
BATCH_SIZE=1
LEARNING_RATE=0.0002
LAMBDA_L1=100

# Intervalos
EVAL_INTERVAL=1000
SAVE_INTERVAL=5000

# Opciones adicionales (descomenta para activar)
# SHOW_SAMPLES="--show-samples"
# NO_AUGMENTATION="--no-augmentation"
# RESTORE_CHECKPOINT="--restore-checkpoint"
```

---

### 2. `predict_model.sh` - Realizar Predicciones

Ejecuta predicciones con un modelo entrenado sobre el dataset de prueba.

**Uso básico:**
```bash
./scripts/predict_model.sh [nombre_modelo] [algoritmo]
```

**Ejemplos:**
```bash
# Predicción con canny_model
./scripts/predict_model.sh canny_model canny

# Predicción con laplaciano_model
./scripts/predict_model.sh laplaciano_model laplaciano
```

**Qué genera:**
- Todas las imágenes predichas en `resultados/{modelo}/imagenes_prueba/`
- Métricas individuales en `resultados/{modelo}/metricas/individual_metrics.txt`
- Resumen de métricas en `resultados/{modelo}/metricas/test_metrics_summary.txt`

---

### 3. `train_all_models.sh` - Entrenar Todos los Modelos

Entrena secuencialmente todos los algoritmos disponibles.

**Uso:**
```bash
./scripts/train_all_models.sh [steps]
```

**Ejemplos:**
```bash
# Entrenar todos con 500k steps
./scripts/train_all_models.sh 500000

# Entrenar todos con configuración por defecto
./scripts/train_all_models.sh
```

**Modelos que se entrenarán:**
- canny_model
- canny_inverso_model
- canny_inverso_erosion_model (canny_inv_erosion_model)
- laplaciano_model
- laplaciano_inverso_erosion_model (laplaciano_inv_erosion_model)

**Características:**
- Pide confirmación antes de iniciar
- Muestra progreso de cada modelo
- Si un modelo falla, pregunta si continuar con el siguiente
- Muestra resumen al final

---

### 4. `predict_all_models.sh` - Predicción con Todos los Modelos

Ejecuta predicciones con todos los modelos entrenados.

**Uso:**
```bash
./scripts/predict_all_models.sh
```

**Características:**
- Evalúa todos los modelos disponibles
- Salta modelos que no tienen pesos guardados
- Genera métricas para cada modelo
- Muestra resumen comparativo al final

**Salida esperada:**
```
============================================================
RESUMEN DE MÉTRICAS
============================================================

--- canny_model ---
PSNR: 28.7891 ± 1.2345
SSIM: 0.8823 ± 0.0234

--- laplaciano_model ---
PSNR: 27.5432 ± 1.5678
SSIM: 0.8654 ± 0.0345
...
```

---

### 5. `quick_test.sh` - Prueba Rápida

Ejecuta un entrenamiento corto para verificar que todo funciona correctamente.

**Uso:**
```bash
./scripts/quick_test.sh [algoritmo]
```

**Ejemplos:**
```bash
# Probar con canny
./scripts/quick_test.sh canny

# Probar con laplaciano
./scripts/quick_test.sh laplaciano
```

**Características:**
- Solo 100 steps (muy rápido)
- Muestra muestras del dataset
- Muestra resumen del modelo
- Útil para debugging

---

## Flujo de Trabajo Recomendado

### 1. Primera vez - Verificar configuración

```bash
# Probar que todo funciona
./scripts/quick_test.sh canny
```

Si no hay errores, continuar.

### 2. Organizar datasets

```bash
python organize.py
```

### 3. Entrenar modelos

**Opción A: Entrenar uno por uno**
```bash
./scripts/train_model.sh canny
./scripts/train_model.sh laplaciano
```

**Opción B: Entrenar todos a la vez**
```bash
./scripts/train_all_models.sh 500000
```

### 4. Monitorear entrenamiento (en otra terminal)

```bash
tensorboard --logdir logs/
```

### 5. Hacer predicciones

**Opción A: Predecir con un modelo**
```bash
./scripts/predict_model.sh canny_model canny
```

**Opción B: Predecir con todos**
```bash
./scripts/predict_all_models.sh
```

### 6. Revisar resultados

```bash
# Ver métricas de un modelo
cat resultados/canny_model/metricas/individual_metrics.txt

# Ver imágenes generadas
ls resultados/canny_model/imagenes_prueba/
```

---

## Modificar Parámetros

### Para modificar parámetros de entrenamiento:

Edita `train_model.sh` y cambia las variables en la sección de CONFIGURACIÓN:

```bash
# Abrir con editor
nano train_model.sh

# Buscar sección "CONFIGURACIÓN"
# Modificar valores según necesites
STEPS=1000000           # Aumentar steps
LEARNING_RATE=0.0001    # Cambiar learning rate
BATCH_SIZE=2            # Aumentar batch size (requiere más memoria)

# Guardar y ejecutar
./scripts/train_model.sh canny
```

### Para modificar la lista de modelos:

Edita `train_all_models.sh` o `predict_all_models.sh`:

```bash
# Abrir con editor
nano train_all_models.sh

# Modificar el array ALGORITMOS
ALGORITMOS=(
    "canny"
    "laplaciano"
    # "canny_inverso"  # Comentar para desactivar
    "mi_nuevo_algoritmo"  # Agregar nuevo
)
```

---

## Opciones Avanzadas

### Activar/Desactivar Data Augmentation

En `train_model.sh`, descomenta la línea:

```bash
NO_AUGMENTATION="--no-augmentation"  # Desactivar augmentation
```

### Continuar desde Checkpoint

En `train_model.sh`, descomenta la línea:

```bash
RESTORE_CHECKPOINT="--restore-checkpoint"  # Continuar entrenamiento
```

### Mostrar Muestras del Dataset

En `train_model.sh`, descomenta la línea:

```bash
SHOW_SAMPLES="--show-samples"  # Ver muestras antes de entrenar
```

### Cambiar Dimensiones de Imagen

En `train_model.sh`, modifica:

```bash
IMG_WIDTH=2048   # Aumentar ancho
IMG_HEIGHT=826   # Aumentar alto
```

---

## Troubleshooting

### Error: "permission denied"

Dale permisos de ejecución:
```bash
chmod +x train_model.sh predict_model.sh train_all_models.sh predict_all_models.sh quick_test.sh
```

### Error: "No se encontró el directorio train"

Primero organiza el dataset:
```bash
python organize.py
```

### Error: "No se encontró el archivo de pesos"

El modelo no está entrenado. Entrénalo primero:
```bash
./scripts/train_model.sh canny
```

### Cambiar configuración en Windows

En Windows, puedes usar Git Bash o WSL para ejecutar los scripts.

**Alternativa:** Crea archivos `.bat` equivalentes:

```batch
@echo off
REM train_model.bat
set ALGORITMO=%1
if "%ALGORITMO%"=="" set ALGORITMO=canny

python train.py ^
    --input-path dataset/%ALGORITMO% ^
    --dataset-name %ALGORITMO%_model ^
    --steps 500000
```

---

## Ejemplos Completos

### Caso 1: Entrenar y evaluar un modelo

```bash
# 1. Organizar dataset
python organize.py

# 2. Prueba rápida
./scripts/quick_test.sh canny

# 3. Entrenar
./scripts/train_model.sh canny canny_model 500000

# 4. Predecir
./scripts/predict_model.sh canny_model canny

# 5. Ver resultados
cat resultados/canny_model/metricas/test_metrics_summary.txt
```

### Caso 2: Entrenar todos los modelos

```bash
# 1. Organizar dataset
python organize.py

# 2. Entrenar todos
./scripts/train_all_models.sh 500000

# 3. Evaluar todos
./scripts/predict_all_models.sh

# 4. Comparar resultados
for model in resultados/*/metricas/test_metrics_summary.txt; do
    echo "=== $model ==="
    tail -n 2 "$model"
done
```

### Caso 3: Experimentar con parámetros

```bash
# Entrenar con diferentes learning rates
# Editar train_model.sh y cambiar LEARNING_RATE

# Experimento 1: LR = 0.0001
./scripts/train_model.sh canny canny_lr1e4 500000

# Experimento 2: LR = 0.0002 (default)
./scripts/train_model.sh canny canny_lr2e4 500000

# Experimento 3: LR = 0.0004
./scripts/train_model.sh canny canny_lr4e4 500000

# Comparar resultados
./scripts/predict_model.sh canny_lr1e4 canny
./scripts/predict_model.sh canny_lr2e4 canny
./scripts/predict_model.sh canny_lr4e4 canny
```

---

## Notas Importantes

1. **Espacio en disco**: Cada modelo requiere ~2-3 GB de espacio
2. **Tiempo de entrenamiento**: 500k steps toman aprox. 8-12 horas en GPU
3. **Memoria GPU**: Batch size 1 requiere ~4-6 GB VRAM
4. **Checkpoints**: Se guardan automáticamente cada 5000 steps
5. **Logs**: TensorBoard logs se guardan en `logs/`

---

## Soporte

Para más información sobre los parámetros disponibles:

```bash
python train.py --help
python predict.py --help
```

Ver documentación completa:
- [README.md](README.md) - Información general
- [QUICKSTART.md](QUICKSTART.md) - Guía rápida
- [ESTRUCTURA_RESULTADOS.md](ESTRUCTURA_RESULTADOS.md) - Estructura de carpetas
