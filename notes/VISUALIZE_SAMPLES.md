# Código para Visualizar Muestras del Dataset en Notebook

## Código para Copiar en el Notebook

Agrega este código después de cargar tus datasets (`train_dataset` y `test_dataset`):

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_dataset_samples(train_dataset, test_dataset):
    """
    Visualiza una muestra de train y test dataset.
    """
    print("="*60)
    print("Visualizando muestras del dataset")
    print("="*60)

    # Obtener una muestra de train
    train_sample = next(iter(train_dataset.take(1)))
    train_input, train_target = train_sample

    # Obtener una muestra de test
    test_sample = next(iter(test_dataset.take(1)))
    test_input, test_target = test_sample

    # Función para desnormalizar de [-1, 1] a [0, 1]
    def denormalize(img):
        return (img + 1) / 2.0

    # Crear figura con 2 filas y 2 columnas
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Fila 1: Train
    axes[0, 0].imshow(denormalize(train_input[0, :, :, 0]), cmap='gray')
    axes[0, 0].set_title('TRAIN - Input Image (Algoritmo)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(denormalize(train_target[0, :, :, 0]), cmap='gray')
    axes[0, 1].set_title('TRAIN - Ground Truth (Original)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Fila 2: Test
    axes[1, 0].imshow(denormalize(test_input[0, :, :, 0]), cmap='gray')
    axes[1, 0].set_title('TEST - Input Image (Algoritmo)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(denormalize(test_target[0, :, :, 0]), cmap='gray')
    axes[1, 1].set_title('TEST - Ground Truth (Original)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Mostrar información
    print("\nInformación de las muestras:")
    print(f"  Train Input shape: {train_input.shape}")
    print(f"  Train Target shape: {train_target.shape}")
    print(f"  Train Input range: [{train_input.numpy().min():.3f}, {train_input.numpy().max():.3f}]")
    print(f"  Train Target range: [{train_target.numpy().min():.3f}, {train_target.numpy().max():.3f}]")
    print(f"\n  Test Input shape: {test_input.shape}")
    print(f"  Test Target shape: {test_target.shape}")
    print(f"  Test Input range: [{test_input.numpy().min():.3f}, {test_input.numpy().max():.3f}]")
    print(f"  Test Target range: [{test_target.numpy().min():.3f}, {test_target.numpy().max():.3f}]")
    print("="*60)

# Llamar la función
visualize_dataset_samples(train_dataset, test_dataset)
```

## Uso en el Notebook

1. **Después de crear los datasets**, agrega la función y llámala:

```python
# Cargar datasets
train_dataset, test_dataset, loader = validate_and_load_dataset(
    input_path='dataset/canny',
    ground_truth_path='dataset/1051 Redimensionadas'
)

# Visualizar muestras
visualize_dataset_samples(train_dataset, test_dataset)
```

## Qué Muestra

La visualización incluye:

### Fila 1: TRAIN
- **Columna izquierda**: Imagen de entrada procesada por el algoritmo (ej: canny, laplaciano)
- **Columna derecha**: Imagen ground truth original (de "1051 Redimensionadas")

### Fila 2: TEST
- **Columna izquierda**: Imagen de entrada procesada por el algoritmo
- **Columna derecha**: Imagen ground truth original

### Información Mostrada:
- Dimensiones de cada imagen (shape)
- Rango de valores de cada imagen
- La imagen se guarda como `dataset_samples.png`

## Interpretación

**Rangos esperados:**
- Las imágenes deberían estar normalizadas en el rango `[-1, 1]`
- Si ves valores fuera de este rango, hay un problema de normalización

**Forma esperada:**
- `(1, 413, 1024, 1)` para batch_size=1
- `(batch, height, width, channels)`

**Visualización:**
- Las imágenes se desnormalizan a `[0, 1]` solo para mostrar
- La imagen Input debe verse como el resultado del algoritmo (bordes, etc.)
- La imagen Ground Truth debe verse como la imagen SEM original

## Ejemplo de Salida

```
============================================================
Visualizando muestras del dataset
============================================================

Información de las muestras:
  Train Input shape: (1, 413, 1024, 1)
  Train Target shape: (1, 413, 1024, 1)
  Train Input range: [-0.998, 0.996]
  Train Target range: [-0.945, 0.987]

  Test Input shape: (1, 413, 1024, 1)
  Test Target shape: (1, 413, 1024, 1)
  Test Input range: [-0.992, 0.998]
  Test Target range: [-0.956, 0.991]
============================================================
```

## Versión Compacta (Si Solo Quieres Ver)

Si solo quieres verificar rápidamente:

```python
# Ver una muestra de train
for input_img, target_img in train_dataset.take(1):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow((input_img[0, :, :, 0] + 1) / 2, cmap='gray')
    plt.title('Train Input')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow((target_img[0, :, :, 0] + 1) / 2, cmap='gray')
    plt.title('Train Ground Truth')
    plt.axis('off')

    plt.show()

    print(f"Input shape: {input_img.shape}, range: [{input_img.numpy().min():.3f}, {input_img.numpy().max():.3f}]")
    print(f"Target shape: {target_img.shape}, range: [{target_img.numpy().min():.3f}, {target_img.numpy().max():.3f}]")
```
