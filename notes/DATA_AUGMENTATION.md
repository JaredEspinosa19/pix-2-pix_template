# Data Augmentation en Pix2Pix

## ¿Qué es Data Augmentation?

El **data augmentation** (aumento de datos) es una técnica que modifica las imágenes de entrenamiento para crear variaciones, ayudando al modelo a generalizar mejor.

## Augmentation en este Código

Por defecto, el código aplica **random jitter** que incluye:

### 1. Random Crop (Recorte Aleatorio)
- Redimensiona la imagen a **1.5x** el tamaño objetivo (ej: de 1024x413 a 1536x620)
- Luego recorta aleatoriamente una región del tamaño original
- **Efecto**: El modelo ve diferentes partes de la imagen en cada época

### 2. Random Flip (Volteo Aleatorio)
- Con 50% de probabilidad, voltea la imagen horizontalmente
- **Efecto**: El modelo aprende que las características pueden aparecer en ambos lados

## Ventajas del Data Augmentation

✅ **Reduce overfitting** - El modelo no memoriza imágenes específicas
✅ **Aumenta la variabilidad** - Más "imágenes virtuales" del mismo dataset
✅ **Mejora generalización** - El modelo aprende características más robustas
✅ **Funciona con datasets pequeños** - Especialmente útil con pocos datos

## Desventajas del Data Augmentation

❌ **Puede perder información de bordes** - Los recortes pueden eliminar partes importantes
❌ **Aumenta tiempo de entrenamiento** - Procesamiento adicional en cada paso
❌ **No siempre necesario** - Si tienes muchos datos variados, puede no ayudar mucho
❌ **Puede introducir artefactos** - En imágenes con estructura específica

## ¿Cuándo Desactivarlo?

Considera desactivar augmentation si:

1. **Dataset grande y variado** - Ya tienes suficiente variabilidad natural
2. **Imágenes con bordes críticos** - Los recortes eliminan información importante
3. **Entrenamiento rápido para pruebas** - Quieres iterar rápidamente
4. **Estructura espacial importante** - La posición exacta de características importa
5. **Imágenes ya pre-procesadas** - Las imágenes ya están optimizadas

## Cómo Usar

### Con Augmentation (Default)

```bash
python train.py \
    --input-path dataset/canny \
    --ground-truth-path "dataset/1051 Redimensionadas"
```

### Sin Augmentation

```bash
python train.py \
    --input-path dataset/canny \
    --ground-truth-path "dataset/1051 Redimensionadas" \
    --no-augmentation
```

## Comparación Visual

### Con Augmentation
```
Imagen Original (1024x413)
         ↓
Resize a 1.5x (1536x620)
         ↓
Random Crop → (1024x413)
         ↓
Random Flip (50% probabilidad)
         ↓
Normalize [-1, 1]
```

### Sin Augmentation
```
Imagen Original (cualquier tamaño)
         ↓
Resize directo a (1024x413)
         ↓
Normalize [-1, 1]
```

## Ejemplo Práctico

### Caso 1: Dataset Pequeño (< 1000 imágenes)
**Recomendación**: Usar augmentation
```bash
python main.py --input-path dataset/canny --steps 500000
```

### Caso 2: Dataset Grande (> 5000 imágenes)
**Recomendación**: Probar sin augmentation primero
```bash
python main.py --input-path dataset/canny --no-augmentation --steps 500000
```

### Caso 3: Imágenes de Bordes (Canny, Laplaciano)
**Recomendación**: Sin augmentation (los bordes en los extremos son importantes)
```bash
python main.py --input-path dataset/canny --no-augmentation --steps 500000
```

### Caso 4: Pruebas Rápidas
**Recomendación**: Sin augmentation (más rápido)
```bash
python main.py --input-path dataset/canny --no-augmentation --steps 10000
```

## En el Notebook

```python
# Con augmentation
train_ds, test_ds, loader = validate_and_load_dataset(
    input_path='dataset/canny',
    ground_truth_path='dataset/1051 Redimensionadas',
    use_augmentation=True  # Default
)

# Sin augmentation
train_ds, test_ds, loader = validate_and_load_dataset(
    input_path='dataset/canny',
    ground_truth_path='dataset/1051 Redimensionadas',
    use_augmentation=False
)
```

## Verificar Augmentation

Usa `--show-samples` para ver el efecto:

```bash
# Con augmentation
python train.py \
    --input-path dataset/canny \
    --show-samples \
    --steps 1

# Sin augmentation
python train.py \
    --input-path dataset/canny \
    --no-augmentation \
    --show-samples \
    --steps 1
```

Ejecuta ambos comandos y compara las imágenes guardadas (`dataset_samples.png`).

**Con augmentation**: Verás que las imágenes pueden estar recortadas y volteadas
**Sin augmentation**: Las imágenes mantienen su estructura completa

## Recomendaciones Específicas para tu Caso

### Para Canny / Laplaciano / Detección de Bordes:

```bash
# RECOMENDADO: Sin augmentation
python train.py \
    --input-path dataset/canny \
    --ground-truth-path "dataset/1051 Redimensionadas" \
    --no-augmentation \
    --steps 500000
```

**Razón**: Los detectores de bordes producen información en toda la imagen, especialmente en los bordes. El random crop puede eliminar bordes importantes.

### Para Imágenes Naturales:

```bash
# RECOMENDADO: Con augmentation
python train.py \
    --input-path dataset/natural_images \
    --ground-truth-path "dataset/1051 Redimensionadas" \
    --steps 500000
```

**Razón**: Las imágenes naturales tienen redundancia y el augmentation ayuda a generalizar.

## Resumen

| Situación | Augmentation | Comando |
|-----------|--------------|---------|
| Dataset pequeño | ✅ ON | (default) |
| Dataset grande | ❓ Probar | `--no-augmentation` |
| Bordes críticos | ❌ OFF | `--no-augmentation` |
| Pruebas rápidas | ❌ OFF | `--no-augmentation` |
| Imágenes naturales | ✅ ON | (default) |
| Canny/Laplaciano | ❌ OFF | `--no-augmentation` |

## Nota Importante

El **test dataset** NUNCA tiene augmentation, independientemente de este flag. Solo afecta al **training dataset**.
