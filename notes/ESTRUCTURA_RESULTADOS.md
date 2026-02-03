# Estructura de Resultados - Pix2Pix

## Organización de Carpetas

Cada modelo entrenado genera su propia estructura de carpetas organizada en `resultados/{nombre_modelo}/`:

```
resultados/{nombre_modelo}/
├── imagenes_entrenamiento/          # Imágenes generadas durante el entrenamiento
│   └── epoch_{step}.jpg            # Comparación: Input | GT | Predicción
│
├── imagenes_prueba/                 # Predicciones sobre dataset de prueba
│   ├── SEM_Imaging_..._s0002.png
│   ├── SEM_Imaging_..._s0003.png
│   └── ...                         # Una imagen por cada archivo del test set
│
├── weights/                         # Todos los pesos del modelo
│   ├── best_generator_weights.h5   # Mejores pesos (basado en PSNR + SSIM)
│   ├── generator_final.keras       # Modelo generador completo (final)
│   └── discriminator_final.keras   # Modelo discriminador completo (final)
│
└── metricas/                        # Métricas de evaluación
    ├── individual_metrics.txt      # Métricas por cada imagen (PSNR y SSIM)
    ├── final_metrics_summary.txt   # Resumen final del entrenamiento
    ├── test_metrics_summary.txt    # Resumen de predicciones (inferencia)
    └── metrics_step_{step}.txt     # Métricas en cada evaluación
```

## Archivos Adicionales

Fuera de la carpeta de resultados:

```
training_checkpoints/      # Checkpoints periódicos de TensorFlow
├── ckpt-1.data-00000-of-00001
├── ckpt-1.index
├── checkpoint
└── ...

logs/                      # Logs de TensorBoard
└── fit/
    └── {timestamp}/
        └── events.out.tfevents...
```

## Descripción Detallada

### 1. imagenes_entrenamiento/

**Contenido**: Imágenes de comparación generadas durante el entrenamiento.

**Formato**: `epoch_{step}.jpg`
- Cada archivo muestra 3 imágenes lado a lado:
  - Input Image (izquierda)
  - Ground Truth (centro)
  - Predicted Image (derecha)

**Frecuencia**: Se generan cada 100,000 pasos (configurable)

**Uso**: Monitorear visualmente el progreso del entrenamiento

---

### 2. imagenes_prueba/

**Contenido**: Todas las predicciones del modelo sobre el dataset de prueba.

**Formato**: Mismo nombre que las imágenes de entrada
- Ejemplo: Si la entrada es `SEM_Imaging_..._s0002.png`, la predicción se guarda con el mismo nombre

**Cuándo se generan**:
- Durante **entrenamiento**: No se generan automáticamente
- Durante **inferencia**: Se generan al ejecutar `predict.py` con `--test-dataset`

**Uso**:
- Comparación visual con ground truth
- Evaluación cualitativa del modelo
- Análisis de casos específicos

---

### 3. weights/

**Contenido**: Todos los pesos y modelos guardados.

#### best_generator_weights.h5
- **Qué es**: Solo los pesos (weights) del generador
- **Criterio**: Mejor combinación de PSNR + SSIM durante entrenamiento
- **Uso principal**: Cargar en inferencia con `--weights`
- **Formato**: HDF5 (.h5)
- **Tamaño**: ~200-500 MB (depende de la arquitectura)

#### generator_final.keras
- **Qué es**: Modelo completo del generador (arquitectura + pesos)
- **Cuándo se guarda**: Al finalizar el entrenamiento
- **Uso**: Cargar modelo completo sin necesidad de definir arquitectura
- **Formato**: Keras (.keras)
- **Tamaño**: Similar a .h5

#### discriminator_final.keras
- **Qué es**: Modelo completo del discriminador
- **Uso**: Continuar entrenamiento o análisis
- **Nota**: No se usa en inferencia (solo se usa el generador)

**Diferencia entre .h5 y .keras**:
- `.h5`: Solo pesos, requiere definir arquitectura antes de cargar
- `.keras`: Modelo completo, incluye arquitectura y pesos

---

### 4. metricas/

**Contenido**: Archivos de texto con métricas cuantitativas.

#### individual_metrics.txt
```
Imagen                                             PSNR       SSIM
------------------------------------------------------------------------
SEM_Imaging_..._s0002.png                        28.3456    0.8765
SEM_Imaging_..._s0003.png                        29.1234    0.8890
...
------------------------------------------------------------------------
Promedio                                          28.7891    0.8823
Desviación estándar                                1.2345    0.0234
```

**Contenido**:
- Nombre de cada imagen del test set
- PSNR individual
- SSIM individual
- Estadísticas agregadas

**Uso**:
- Identificar imágenes con bajo rendimiento
- Análisis de casos específicos
- Verificar consistencia del modelo

#### final_metrics_summary.txt
```
Métricas Finales del Entrenamiento
============================================================

Total de imágenes: 210

PSNR: 28.7891 ± 1.2345
SSIM: 0.8823 ± 0.0234
```

**Cuándo se genera**: Al finalizar el entrenamiento (main.py)

**Contenido**:
- Total de imágenes evaluadas
- PSNR promedio ± desviación estándar
- SSIM promedio ± desviación estándar

#### test_metrics_summary.txt
```
Resumen de Métricas del Dataset de Prueba
============================================================

Total de imágenes: 210

PSNR: 28.7891 ± 1.2345
SSIM: 0.8823 ± 0.0234
```

**Cuándo se genera**: Durante inferencia con `predict.py --test-dataset`

**Contenido**: Similar a final_metrics_summary.txt pero para inferencia

#### metrics_step_{step}.txt
```
Step: 50000
PSNR: 27.5432
SSIM: 0.8654
Score: 36.4086
```

**Cuándo se genera**: Cada vez que se guardan mejores pesos durante entrenamiento

**Contenido**:
- Paso de entrenamiento
- PSNR en ese momento
- SSIM en ese momento
- Score (PSNR + SSIM) usado para determinar "mejor modelo"

**Uso**: Rastrear la evolución de las métricas durante entrenamiento

---

## Flujo de Trabajo Típico

### Durante Entrenamiento

1. **Paso 1-500,000**: Entrenamiento
   - Se guardan checkpoints cada 5,000 pasos en `training_checkpoints/`
   - Se evalúa cada 1,000 pasos
   - Si las métricas mejoran → se guarda `best_generator_weights.h5`
   - Se generan imágenes cada 100,000 pasos en `imagenes_entrenamiento/`

2. **Al finalizar**:
   - Se guardan `generator_final.keras` y `discriminator_final.keras` en `weights/`
   - Se calculan métricas finales y se guardan en `metricas/individual_metrics.txt` y `final_metrics_summary.txt`

### Durante Inferencia

```bash
python predict.py \
    --weights resultados/canny_model/weights/best_generator_weights.h5 \
    --test-dataset dataset/canny \
    --output-dir resultados/canny_model
```

1. Carga los pesos del generador
2. Procesa cada imagen del test set
3. Guarda predicciones en `imagenes_prueba/`
4. Calcula métricas individuales y resumen
5. Guarda `individual_metrics.txt` y `test_metrics_summary.txt` en `metricas/`

---

## Ejemplos de Uso

### Ver métricas de una imagen específica

```bash
# Linux/Mac
grep "SEM_Imaging_..._s0002.png" resultados/canny_model/metricas/individual_metrics.txt

# Windows
findstr "SEM_Imaging_..._s0002.png" resultados\canny_model\metricas\individual_metrics.txt
```

### Encontrar imágenes con bajo PSNR

Abrir `individual_metrics.txt` y buscar valores de PSNR < 25 dB

### Comparar modelos diferentes

```bash
# Ver resumen de canny
cat resultados/canny_model/metricas/final_metrics_summary.txt

# Ver resumen de laplaciano
cat resultados/laplaciano_model/metricas/final_metrics_summary.txt
```

### Continuar entrenamiento desde checkpoint

```bash
python train.py \
    --input-path dataset/canny \
    --dataset-name canny_model \
    --restore-checkpoint
```

Esto restaura desde `training_checkpoints/` y continúa entrenando.

---

## Notas Importantes

1. **Espacio en Disco**:
   - `weights/`: ~500 MB por modelo
   - `imagenes_prueba/`: ~100-200 MB (210 imágenes PNG)
   - `training_checkpoints/`: ~1-2 GB
   - `logs/`: ~100-500 MB

2. **Archivos Esenciales**:
   - Para inferencia: `weights/best_generator_weights.h5`
   - Para análisis: `metricas/individual_metrics.txt`
   - Para continuar entrenamiento: `training_checkpoints/`

3. **Archivos Opcionales**:
   - `discriminator_final.keras`: Solo si quieres continuar entrenamiento
   - `imagenes_entrenamiento/`: Solo para visualización
   - `logs/`: Solo para TensorBoard

4. **Limpieza**:
   Si necesitas espacio, puedes eliminar:
   - `training_checkpoints/` (después de finalizar entrenamiento)
   - `logs/` (después de revisar en TensorBoard)
   - `imagenes_entrenamiento/` (si ya verificaste el entrenamiento)

   **NO elimines**:
   - `weights/`
   - `metricas/`
   - `imagenes_prueba/`
