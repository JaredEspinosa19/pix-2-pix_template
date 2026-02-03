# Fix para el Error de Dimensiones en el Notebook

## Problema

Cuando ejecutas el entrenamiento en el notebook, obtienes este error:

```
ValueError: Input 0 with name 'input_layer_40' of layer 'functional_57' is incompatible with the layer:
expected shape=(None, 413, 1024, 1), found shape=(1, 1, 413, 1024)
```

## Causa

El error ocurre porque el dataset ya retorna **batches** (debido a `.batch(BATCH_SIZE)`), entonces cada `input_image` ya tiene forma `(1, 413, 1024, 1)`.

Cuando haces `tf.expand_dims(input_image, 0)`, estás agregando otra dimensión de batch, resultando en `(1, 1, 413, 1024)`, lo cual es incorrecto.

## Solución

### 1. Actualizar la función `evaluate_model`

**ANTES (❌ INCORRECTO):**
```python
def evaluate_model(generator, test_dataset, save_dir=None):
    psnr_values = []
    ssim_values = []

    for idx, (input_image, target) in enumerate(tqdm(test_dataset)):
        prediction = generator(tf.expand_dims(input_image, 0), training=False)[0]  # ❌ Error aquí

        psnr = tf.image.psnr(target, prediction, max_val=1.0).numpy()
        ssim = tf.image.ssim(target, prediction, max_val=1.0).numpy()

        psnr_values.append(psnr)
        ssim_values.append(ssim)

    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)

    return mean_psnr, mean_ssim
```

**DESPUÉS (✅ CORRECTO):**
```python
def evaluate_model(generator, test_dataset, save_dir=None):
    psnr_values = []
    ssim_values = []

    for idx, (input_image, target) in enumerate(tqdm(test_dataset)):
        # input_image ya tiene forma (1, 413, 1024, 1) del batch
        prediction = generator(input_image, training=False)  # ✅ Sin expand_dims

        # max_val=2.0 porque las imágenes están normalizadas en [-1, 1]
        psnr = tf.image.psnr(target, prediction, max_val=2.0).numpy()
        ssim = tf.image.ssim(target, prediction, max_val=2.0).numpy()

        # Extraer el valor escalar del array
        psnr_values.append(psnr[0])
        ssim_values.append(ssim[0])

    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)

    return mean_psnr, mean_ssim
```

### 2. Actualizar la función `fit` (si usas evaluate_model allí)

Si tu función `fit` llama a `evaluate_model`, asegúrate de que no esté haciendo `expand_dims` tampoco.

### 3. Actualizar `generate_images` si es necesario

Si tu función `generate_images` también tiene este problema:

**ANTES:**
```python
def generate_images(model, test_input, tar, ...):
    prediction = model(tf.expand_dims(test_input, 0), training=True)  # ❌
```

**DESPUÉS:**
```python
def generate_images(model, test_input, tar, ...):
    prediction = model(test_input, training=True)  # ✅
```

## Cambios Clave

1. **Eliminar `tf.expand_dims(input_image, 0)`** - Ya no es necesario
2. **Cambiar `max_val=1.0` a `max_val=2.0`** - Porque las imágenes están en rango [-1, 1]
3. **Extraer valores con `[0]`** - `psnr[0]` y `ssim[0]` para obtener valores escalares

## ¿Por qué max_val=2.0?

Las imágenes son normalizadas así:
```python
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1  # Rango: [-1, 1]
    real_image = (real_image / 127.5) - 1    # Rango: [-1, 1]
    return input_image, real_image
```

- Valor mínimo: -1
- Valor máximo: 1
- Rango total: 1 - (-1) = **2**

Por lo tanto, `max_val=2.0` es correcto para calcular PSNR y SSIM.

## Verificación

Para verificar que las dimensiones son correctas, puedes agregar un print:

```python
for input_image, target in test_dataset:
    print(f"Input shape: {input_image.shape}")  # Debe ser (1, 413, 1024, 1)
    print(f"Target shape: {target.shape}")       # Debe ser (1, 413, 1024, 1)
    prediction = generator(input_image, training=False)
    print(f"Prediction shape: {prediction.shape}") # Debe ser (1, 413, 1024, 1)
    break
```

## Archivos Actualizados

Los archivos de Python ya han sido actualizados con estos cambios:
- ✅ `training.py` - Función `evaluate_model` corregida
- ✅ `training.py` - Función `calculate_metrics` corregida

Ahora solo necesitas aplicar estos cambios en tu notebook.