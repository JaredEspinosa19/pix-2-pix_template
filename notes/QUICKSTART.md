# Guía Rápida - Pix2Pix

## 1. Organizar el Dataset

Si tus imágenes no están organizadas en train/test, ejecuta:

```bash
python organize.py
```

Esto organizará automáticamente las imágenes según los archivos `dataset/images_list/*.txt`

## 2. Entrenar un Modelo

### Verificar Dataset (Recomendado)

Antes de entrenar, verifica que el dataset se esté cargando correctamente:

```bash
python train.py \
    --input-path dataset/canny \
    --ground-truth-path "dataset/1051 Redimensionadas" \
    --show-samples \
    --steps 1
```

Esto mostrará imágenes de muestra y creará `dataset_samples.png`. Verifica que:
- Las imágenes input muestren el resultado del algoritmo (bordes, etc.)
- Las imágenes ground truth muestren la imagen SEM original
- Los rangos de valores estén en [-1, 1]

### Para Canny:
```bash
python train.py \
    --input-path dataset/canny \
    --ground-truth-path "dataset/1051 Redimensionadas" \
    --dataset-name canny_model \
    --steps 500000
```

### Para Canny Inverso:
```bash
python train.py \
    --input-path dataset/canny_inverso \
    --ground-truth-path "dataset/1051 Redimensionadas" \
    --dataset-name canny_inverso_model \
    --steps 500000
```

### Para Canny Inverso + Erosión:
```bash
python train.py \
    --input-path dataset/canny_inverso_erosion \
    --ground-truth-path "dataset/1051 Redimensionadas" \
    --dataset-name canny_inv_erosion_model \
    --steps 500000
```

### Para Laplaciano:
```bash
python train.py \
    --input-path dataset/laplaciano \
    --ground-truth-path "dataset/1051 Redimensionadas" \
    --dataset-name laplaciano_model \
    --steps 500000
```

### Para Laplaciano Inverso + Erosión:
```bash
python train.py \
    --input-path dataset/laplaciano_inverso_erosion \
    --ground-truth-path "dataset/1051 Redimensionadas" \
    --dataset-name laplaciano_inv_erosion_model \
    --steps 500000
```

## 3. Monitorear Entrenamiento

Durante el entrenamiento, puedes ver las métricas en TensorBoard:

```bash
tensorboard --logdir logs/
```

Abre tu navegador en: http://localhost:6006

## 4. Hacer Predicciones

### Predecir una imagen:
```bash
python predict.py \
    --weights resultados/canny_model/weights/best_generator_weights.h5 \
    --input-image test_image.png \
    --output-dir resultados/canny_model/predicciones \
    --visualize
```

### Predecir todas las imágenes de test:
```bash
python predict.py \
    --weights resultados/canny_model/weights/best_generator_weights.h5 \
    --test-dataset dataset/canny \
    --ground-truth-path "dataset/1051 Redimensionadas" \
    --output-dir resultados/canny_model
```

## Archivos Generados

Después del entrenamiento encontrarás todo organizado en `resultados/{dataset_name}/`:

```
resultados/{dataset_name}/
├── imagenes_entrenamiento/          # Imágenes generadas durante entrenamiento
├── imagenes_prueba/                 # Todas las predicciones del test set
│   ├── {imagen1}.png
│   ├── {imagen2}.png
│   └── ...
├── weights/                         # Pesos del modelo
│   ├── best_generator_weights.h5   # Mejores pesos (usar para inferencia)
│   ├── generator_final.keras
│   └── discriminator_final.keras
└── metricas/                        # Métricas de evaluación
    ├── individual_metrics.txt      # PSNR y SSIM por cada imagen
    ├── final_metrics_summary.txt   # Resumen del entrenamiento
    └── test_metrics_summary.txt    # Resumen de inferencia
```

Adicionalmente:
- `training_checkpoints/` - Checkpoints periódicos
- `logs/` - Logs de TensorBoard

**Ver métricas**: Abre `metricas/individual_metrics.txt` para ver PSNR y SSIM de cada imagen.

## Parámetros Importantes

- `--steps`: Número de pasos de entrenamiento (default: 500000)
- `--eval-interval`: Cada cuántos pasos evaluar (default: 1000)
- `--save-interval`: Cada cuántos pasos guardar checkpoint (default: 5000)
- `--batch-size`: Tamaño del batch (default: 1)
- `--learning-rate`: Tasa de aprendizaje (default: 2e-4)
- `--lambda-l1`: Peso de la pérdida L1 (default: 100)
- `--no-augmentation`: Desactivar data augmentation (random crop y flip)

## Tips

1. **GPU**: El código detecta automáticamente si hay GPU disponible
2. **Checkpoints**: Usa `--restore-checkpoint` para continuar entrenamiento
3. **Evaluación**: El modelo guarda automáticamente los mejores pesos según PSNR + SSIM
4. **Memoria**: Si tienes problemas de memoria, reduce `--batch-size`

## Ejemplo Completo

```bash
# 1. Organizar dataset
python organize.py

# 2. Entrenar modelo
python train.py \
    --input-path dataset/canny \
    --ground-truth-path "dataset/1051 Redimensionadas" \
    --dataset-name canny_model \
    --steps 500000 \
    --eval-interval 1000

# 3. En otra terminal, monitorear
tensorboard --logdir logs/

# 4. Hacer predicciones
python predict.py \
    --weights resultados/canny_model/weights/best_generator_weights.h5 \
    --test-dataset dataset/canny \
    --ground-truth-path "dataset/1051 Redimensionadas" \
    --output-dir resultados/canny_model
```
