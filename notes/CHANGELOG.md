# Registro de Cambios - Reorganización del Proyecto

## Última Actualización: Scripts de Bash Organizados

**Fecha**: Febrero 2026

### Cambios:
- ✅ Todos los scripts bash movidos a carpeta `scripts/`
- ✅ Documentación movida a `scripts/README.md`
- ✅ README principal actualizado con nuevas rutas
- ✅ Permisos de ejecución mantenidos

**Nuevas rutas:**
```bash
./scripts/train_model.sh         # Antes: ./train_model.sh
./scripts/predict_model.sh       # Antes: ./predict_model.sh
./scripts/train_all_models.sh    # Antes: ./train_all_models.sh
./scripts/predict_all_models.sh  # Antes: ./predict_all_models.sh
./scripts/quick_test.sh          # Antes: ./quick_test.sh
```

---

## Cambios Realizados Anteriormente

### 1. Reorganización de Código

**Estructura Anterior:**
```
pix-2-pix_template/
├── dataset_loader.py
├── network.py
├── training.py
├── main.py
├── inference.py
└── organize_dataset.py
```

**Nueva Estructura:**
```
pix-2-pix_template/
├── src/                        # Todo el código fuente
│   ├── dataset_loader.py
│   ├── network.py
│   ├── training.py
│   ├── main.py
│   ├── inference.py
│   └── organize_dataset.py
├── train.py                    # Script de entrada para entrenar
├── predict.py                  # Script de entrada para inferencia
└── organize.py                 # Script de entrada para organizar dataset
```

### 2. Nueva Estructura de Resultados

**Antes:**
- Archivos dispersos en la raíz: `{nombre}_best_generator_weights.h5`, etc.
- Carpeta `resultados/{nombre}/` sin estructura clara

**Ahora:**
```
resultados/{nombre_modelo}/
├── imagenes_entrenamiento/     # Imágenes generadas durante entrenamiento
│   └── epoch_{step}.jpg
├── imagenes_prueba/             # Predicciones en dataset de prueba
├── weights/                     # Todos los pesos del modelo
│   ├── best_generator_weights.h5
│   ├── generator_final.keras
│   └── discriminator_final.keras
└── metricas/                    # Métricas de evaluación
    ├── metrics_step_{step}.txt
    └── final_metrics.txt
```

### 3. Comandos Actualizados

**Entrenamiento:**
```bash
# Antes
python main.py --input-path dataset/canny

# Ahora
python train.py --input-path dataset/canny
```

**Inferencia:**
```bash
# Antes
python inference.py --weights canny_model_best_generator_weights.h5

# Ahora
python predict.py --weights resultados/canny_model/weights/best_generator_weights.h5
```

**Organizar Dataset:**
```bash
# Antes
python organize_dataset.py

# Ahora
python organize.py
```

### 4. Cambios en el Código

#### training.py
- Estructura de carpetas organizada en `__init__`
- Guardado automático de métricas en `metricas/`
- Pesos guardados en `weights/`
- Imágenes de entrenamiento en `imagenes_entrenamiento/`
- Imágenes de prueba en `imagenes_prueba/`

#### inference.py
- Predicciones guardadas en `imagenes_prueba/`
- Métricas guardadas en `metricas/`

#### main.py
- Guardado de métricas finales en la estructura correcta
- Eliminado guardado de modelos en la raíz

### 5. Scripts de Entrada

Los nuevos scripts de entrada (`train.py`, `predict.py`, `organize.py`):
- Agregan automáticamente `src/` al path de Python
- Llaman a las funciones principales de los módulos correspondientes
- Simplifican la ejecución para el usuario

### 6. Documentación Actualizada

- **README.md**: Actualizado con nueva estructura y comandos
- **QUICKSTART.md**: Actualizado con nuevos scripts y rutas
- **DATA_AUGMENTATION.md**: Actualizado con comando `train.py`
- **.gitignore**: Mejorado para incluir archivos generados

## Ventajas de los Cambios

1. **Organización Clara**: Todo el código fuente en `src/`, fácil de mantener
2. **Resultados Estructurados**: Cada modelo tiene su propia carpeta organizada
3. **Scripts Simples**: `train.py`, `predict.py`, `organize.py` son intuitivos
4. **Mejor Modularidad**: Separación clara entre código y scripts de entrada
5. **Fácil Navegación**: Estructura de carpetas predecible y consistente

## Compatibilidad

- ✅ Los argumentos de línea de comandos permanecen iguales
- ✅ La funcionalidad interna no cambió
- ✅ Solo cambiaron los nombres de los scripts de entrada
- ✅ La estructura de carpetas de resultados es nueva pero más organizada

## Migración

Si tenías entrenamiento previo con la estructura antigua:

1. Los archivos en la raíz (`*_best_generator_weights.h5`, etc.) siguen siendo válidos
2. Puedes moverlos manualmente a `resultados/{nombre}/weights/`
3. Los checkpoints en `training_checkpoints/` siguen funcionando
4. Los logs en `logs/` siguen siendo compatibles con TensorBoard
