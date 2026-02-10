# Pix2Pix - Código Modular

Este proyecto implementa una red Pix2Pix (Image-to-Image Translation) organizada en módulos para facilitar su uso y mantenimiento.

## Estructura del Proyecto

```
pix-2-pix_template/
│
├── src/                        # Código fuente
│   ├── dataset_loader.py       # Validación y carga del dataset
│   ├── network.py              # Arquitectura de la red (Generator, Discriminator)
│   ├── training.py             # Lógica de entrenamiento y evaluación
│   ├── main.py                 # Script principal para entrenar
│   ├── inference.py            # Script para realizar predicciones
│   └── organize_dataset.py     # Script para organizar imágenes en train/test
│
├── train.py                    # Script de entrada para entrenar
├── predict.py                  # Script de entrada para inferencia
├── organize.py                 # Script de entrada para organizar dataset
├── run_script.sh               # 🔧 Wrapper para ejecutar scripts fácilmente
│
├── scripts/                    # 🔧 Scripts de bash para automatización
│   ├── train_model.sh          # Entrenar un modelo individual
│   ├── predict_model.sh        # Realizar predicciones
│   ├── train_all_models.sh     # Entrenar todos los modelos
│   ├── predict_all_models.sh   # Predecir con todos los modelos
│   ├── quick_test.sh           # Prueba rápida
│   └── README.md               # Guía de uso de scripts
│
├── dataset/                    # Datasets
├── resultados/                 # Resultados de entrenamiento e inferencia
│   └── {nombre_modelo}/
│       ├── imagenes_entrenamiento/  # Imágenes generadas durante entrenamiento
│       ├── imagenes_prueba/         # Predicciones en test
│       ├── weights/                 # Pesos del modelo
│       └── metricas/                # Métricas de evaluación
│
└── README.md                   # Este archivo
```

## Módulos

### 1. dataset_loader.py
**Funcionalidad:** Validación y carga del dataset

- `DatasetValidator`: Valida que el dataset tenga la estructura correcta
  - Verifica carpetas `train/` y `test/`
  - Cuenta imágenes .png
  - Valida formato de imágenes

- `DatasetLoader`: Carga y preprocesa imágenes
  - Carga imágenes concatenadas (entrada|objetivo)
  - Aplica data augmentation para entrenamiento
  - Normaliza al rango [-1, 1]
  - Crea datasets de TensorFlow

### 2. network.py
**Funcionalidad:** Arquitectura de la red

- `Generator()`: Red U-Net generadora
  - Encoder con 8 capas de downsampling
  - Decoder con skip connections
  - Salida con activación tanh

- `Discriminator()`: Discriminador PatchGAN
  - Clasifica patches como reales o generados
  - Arquitectura convolucional

- `Pix2PixLoss`: Funciones de pérdida
  - Pérdida del generador (GAN + L1)
  - Pérdida del discriminador

### 3. training.py
**Funcionalidad:** Entrenamiento, evaluación y métricas

- `Pix2PixTrainer`: Clase principal de entrenamiento
  - Train step con GradientTape
  - Evaluación con PSNR y SSIM
  - Guardado de checkpoints
  - Integración con TensorBoard

- `Pix2PixInference`: Clase para inferencia
  - Carga de modelos entrenados
  - Predicción sobre imágenes individuales o directorios
  - Visualización de resultados

## Inicio Rápido

### Opción 1: Usando Scripts (Recomendado) 🚀

Los scripts facilitan el entrenamiento y predicción con configuración sencilla.

**En Windows (CMD/PowerShell):**
```cmd
REM 1. Organizar dataset
python organize.py

REM 2. Prueba rápida
scripts\quick_test.bat canny

REM 3. Entrenar modelo
scripts\train_model.bat canny

REM 4. Hacer predicciones
scripts\predict_model.bat canny_model canny
```

**En Linux/Mac (Bash):**
```bash
# 1. Organizar dataset
python organize.py

# 2. Prueba rápida
./scripts/quick_test.sh canny

# 3. Entrenar modelo
./scripts/train_model.sh canny

# 4. Hacer predicciones
./scripts/predict_model.sh canny_model canny
```

**Ver guías completas**:
- Windows: [scripts/WINDOWS.md](scripts/WINDOWS.md) 🪟
- Linux/Mac: [scripts/README.md](scripts/README.md) 🐧

### Opción 2: Usando Python Directamente

## Organizar Dataset

Si tus imágenes no están organizadas en carpetas train/test, usa el script `organize.py`:

```bash
python organize.py
```

Este script:
- Lee los archivos `dataset/images_list/train_images.txt` y `test_images.txt`
- Organiza automáticamente las imágenes en carpetas train/test para cada algoritmo
- Las imágenes se **copian** (no se mueven) a las carpetas correspondientes

## Uso con Python

### Estructura del Dataset

El dataset debe tener la siguiente estructura:

```
dataset/
├── 1051 Redimensionadas/          # Imágenes ground truth
│   ├── SEM Imaging_..._s0002.png
│   ├── SEM Imaging_..._s0003.png
│   └── ...
├── canny/                          # Imágenes de entrada (algoritmo canny)
│   ├── train/
│   │   ├── SEM Imaging_..._s0004.png
│   │   ├── SEM Imaging_..._s0006.png
│   │   └── ...
│   └── test/
│       ├── SEM Imaging_..._s0002.png
│       ├── SEM Imaging_..._s0003.png
│       └── ...
├── laplaciano/                     # Otro algoritmo
│   ├── train/
│   └── test/
└── images_list/                    # Listas de división train/test
    ├── train_images.txt
    └── test_images.txt
```

**Importante:**
- Las imágenes de entrada y ground truth deben tener el **mismo nombre de archivo**
- El ground truth está en una carpeta separada (`1051 Redimensionadas`)
- Cada algoritmo (canny, laplaciano, etc.) tiene sus propias carpetas train/test

### 1. Entrenar un Modelo

#### Uso básico:
```bash
python train.py \
    --input-path dataset/canny \
    --ground-truth-path "dataset/1051 Redimensionadas"
```

#### Con visualización de muestras:
```bash
python train.py \
    --input-path dataset/canny \
    --ground-truth-path "dataset/1051 Redimensionadas" \
    --show-samples
```

Esto mostrará una imagen de muestra de train y test (input y ground truth) antes de comenzar el entrenamiento.

#### Configuración completa:
```bash
python train.py \
    --input-path dataset/canny \
    --ground-truth-path "dataset/1051 Redimensionadas" \
    --dataset-name canny_model \
    --steps 500000 \
    --batch-size 1 \
    --learning-rate 2e-4 \
    --lambda-l1 100 \
    --eval-interval 1000 \
    --save-interval 5000 \
    --img-width 1024 \
    --img-height 413
```

#### Argumentos disponibles:

**Dataset:**
- `--input-path`: Ruta a la carpeta con imágenes de entrada [requerido]
- `--ground-truth-path`: Ruta a las imágenes ground truth (default: dataset/1051 Redimensionadas)
- `--dataset-name`: Nombre para guardar resultados
- `--img-width`: Ancho de las imágenes (default: 1024)
- `--img-height`: Alto de las imágenes (default: 413)

**Entrenamiento:**
- `--steps`: Número total de pasos (default: 500000)
- `--batch-size`: Tamaño del batch (default: 1)
- `--learning-rate`: Tasa de aprendizaje (default: 2e-4)
- `--beta-1`: Parámetro beta_1 de Adam (default: 0.5)
- `--lambda-l1`: Peso de la pérdida L1 (default: 100)

**Evaluación:**
- `--eval-interval`: Intervalo para evaluar (default: 1000)
- `--save-interval`: Intervalo para guardar checkpoints (default: 5000)
- `--checkpoint-dir`: Directorio de checkpoints (default: ./training_checkpoints)
- `--log-dir`: Directorio de logs (default: ./logs)

**Opcionales:**
- `--restore-checkpoint`: Restaurar desde último checkpoint
- `--show-model-summary`: Mostrar resumen de los modelos
- `--show-samples`: Mostrar imágenes de muestra del dataset antes de entrenar
- `--no-augmentation`: Desactivar data augmentation (random crop y flip)

### 2. Realizar Predicciones

#### Predecir una imagen individual:
```bash
python predict.py \
    --weights resultados/canny_model/weights/best_generator_weights.h5 \
    --input-image test_image.png \
    --output-dir resultados/canny_model/predicciones \
    --visualize
```

#### Predecir un directorio completo:
```bash
python predict.py \
    --weights resultados/canny_model/weights/best_generator_weights.h5 \
    --input-dir dataset/test \
    --output-dir resultados/canny_model/predicciones
```

#### Evaluar en dataset de prueba:
```bash
python predict.py \
    --weights resultados/canny_model/weights/best_generator_weights.h5 \
    --test-dataset dataset/canny \
    --ground-truth-path "dataset/1051 Redimensionadas" \
    --output-dir resultados/canny_model
```

#### Argumentos disponibles:

**Modelo:**
- `--weights`: Ruta al archivo de pesos (.h5 o .keras) [requerido]
- `--img-width`: Ancho de las imágenes (default: 1024)
- `--img-height`: Alto de las imágenes (default: 413)

**Entrada:**
- `--input-image`: Imagen individual para predecir
- `--input-dir`: Directorio con imágenes
- `--test-dataset`: Dataset de prueba (carpeta con train/test, para métricas)
- `--ground-truth-path`: Ruta a las imágenes ground truth (default: dataset/1051 Redimensionadas)

**Salida:**
- `--output-dir`: Directorio de salida (default: predictions)
- `--visualize`: Visualizar predicciones

### 3. Usar los Módulos en Código Python

```python
import sys
sys.path.insert(0, 'src')

from dataset_loader import validate_and_load_dataset
from network import create_pix2pix_model, create_optimizers
from training import Pix2PixTrainer

# 1. Cargar dataset
train_ds, test_ds, loader = validate_and_load_dataset(
    input_path='dataset/canny',
    ground_truth_path='dataset/1051 Redimensionadas',
    img_width=1024,
    img_height=413
)

# 2. Crear modelo
generator, discriminator, loss_fn = create_pix2pix_model(
    img_height=413,
    img_width=1024,
    lambda_l1=100
)

# 3. Crear optimizadores
gen_opt, disc_opt = create_optimizers(learning_rate=2e-4)

# 4. Entrenar
trainer = Pix2PixTrainer(
    generator=generator,
    discriminator=discriminator,
    loss_fn=loss_fn,
    generator_optimizer=gen_opt,
    discriminator_optimizer=disc_opt,
    dataset_name='canny_model'
)

trainer.fit(train_ds, test_ds, steps=500000)
```

## Monitoreo con TensorBoard

Durante el entrenamiento, puedes monitorear las métricas en tiempo real:

```bash
tensorboard --logdir logs/
```

Esto mostrará:
- Pérdida total del generador
- Pérdida GAN del generador
- Pérdida L1 del generador
- Pérdida del discriminador

## Archivos Generados

Durante el entrenamiento se genera la siguiente estructura:

```
resultados/{dataset_name}/
├── imagenes_entrenamiento/          # Imágenes generadas durante entrenamiento
│   └── epoch_{step}.jpg            # Comparación: Input | GT | Predicción
│
├── imagenes_prueba/                 # Predicciones sobre dataset de prueba
│   ├── {imagen1}.png               # Todas las imágenes generadas
│   ├── {imagen2}.png
│   └── ...
│
├── weights/                         # Pesos del modelo
│   ├── best_generator_weights.h5   # Mejores pesos (PSNR + SSIM)
│   ├── generator_final.keras       # Modelo generador completo
│   └── discriminator_final.keras   # Modelo discriminador completo
│
└── metricas/                        # Métricas de evaluación
    ├── individual_metrics.txt      # Métricas por cada imagen (PSNR, SSIM)
    ├── final_metrics_summary.txt   # Resumen final del entrenamiento
    ├── test_metrics_summary.txt    # Resumen de inferencia
    └── metrics_step_{step}.txt     # Métricas en cada evaluación
```

Adicionalmente:
- `training_checkpoints/`: Checkpoints periódicos del modelo
- `logs/`: Logs de TensorBoard

**Ver detalles**: [ESTRUCTURA_RESULTADOS.md](ESTRUCTURA_RESULTADOS.md) para una explicación completa de cada archivo.

## Requisitos

Las dependencias se encuentran en `requirements_lightning.txt`:

```bash
pip install -r requirements_lightning.txt
```

Principales librerías:
- TensorFlow
- NumPy
- Matplotlib
- Pillow
- IPython

## GPU

El código detecta automáticamente si hay GPUs disponibles y las utiliza para acelerar el entrenamiento.

## Notas

- Las imágenes deben estar en formato PNG
- El código espera imágenes en escala de grises (1 canal)
- Las dimensiones por defecto son 1024x413, pero se pueden ajustar
- El modelo guarda automáticamente los mejores pesos basándose en PSNR + SSIM