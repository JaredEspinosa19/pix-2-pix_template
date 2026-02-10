# Guía para Ejecutar en Windows

Esta guía explica cómo ejecutar los scripts de entrenamiento y predicción en Windows.

## 📋 Requisitos

- Python instalado (verifica con `python --version`)
- TensorFlow y dependencias instaladas (ver `requirements.txt`)
- CMD, PowerShell o Git Bash

## 🚀 Métodos de Ejecución

### Método 1: Scripts .bat (Nativo de Windows) ⭐ Recomendado

Los scripts `.bat` funcionan nativamente en Windows sin necesidad de Git Bash o WSL.

**Desde CMD o PowerShell:**

```cmd
REM Prueba rápida
scripts\quick_test.bat canny

REM Entrenar modelo
scripts\train_model.bat canny

REM Entrenar con parámetros personalizados
scripts\train_model.bat canny canny_model 500000

REM Hacer predicciones
scripts\predict_model.bat canny_model canny
```

**Características:**
- ✅ Funciona en CMD y PowerShell
- ✅ No requiere Git Bash ni WSL
- ✅ Sintaxis nativa de Windows
- ✅ Cambio automático al directorio raíz

---

### Método 2: Git Bash (Si tienes Git instalado)

Si tienes Git para Windows instalado, puedes usar Git Bash:

```bash
# Abrir Git Bash en la carpeta del proyecto

# Prueba rápida
./scripts/quick_test.sh canny

# Entrenar modelo
./scripts/train_model.sh canny

# Hacer predicciones
./scripts/predict_model.sh canny_model canny
```

**Características:**
- ✅ Usa los scripts originales .sh
- ✅ Sintaxis Unix/Linux
- ⚠️ Requiere Git para Windows

---

### Método 3: Python Directamente

Ejecuta directamente el código Python sin scripts:

**Entrenar:**
```cmd
python train.py --input-path dataset/canny --ground-truth-path "dataset/1051 Redimensionadas" --dataset-name canny_model --steps 500000
```

**Predecir:**
```cmd
python predict.py --weights resultados/canny_model/weights/best_generator_weights.h5 --test-dataset dataset/canny --output-dir resultados/canny_model
```

**Características:**
- ✅ Funciona en cualquier terminal
- ✅ No requiere scripts
- ⚠️ Comando más largo
- ⚠️ Necesitas recordar todos los parámetros

---

### Método 4: WSL (Windows Subsystem for Linux)

Si tienes WSL instalado:

```bash
# En WSL, navegar al proyecto
cd /mnt/c/Users/choc-/OneDrive/Documents/Proyectos\ DMREF/pix-2-pix_template/

# Usar scripts bash normales
./scripts/train_model.sh canny
```

**Características:**
- ✅ Entorno Linux completo
- ✅ Usa scripts .sh
- ⚠️ Requiere WSL instalado y configurado

---

## 📝 Ejemplos Completos

### Ejemplo 1: Flujo Completo con .bat

```cmd
REM 1. Abrir CMD o PowerShell en la carpeta del proyecto
cd C:\Users\choc-\OneDrive\Documents\Proyectos DMREF\pix-2-pix_template

REM 2. Organizar dataset (solo primera vez)
python organize.py

REM 3. Prueba rápida (opcional)
scripts\quick_test.bat canny

REM 4. Entrenar modelo
scripts\train_model.bat canny

REM 5. En otra ventana, monitorear con TensorBoard
tensorboard --logdir logs

REM 6. Hacer predicciones
scripts\predict_model.bat canny_model canny

REM 7. Ver resultados
type resultados\canny_model\metricas\test_metrics_summary.txt
```

### Ejemplo 2: Entrenar Todos los Algoritmos

```cmd
REM Entrenar canny
scripts\train_model.bat canny

REM Entrenar laplaciano
scripts\train_model.bat laplaciano

REM Entrenar canny_inverso
scripts\train_model.bat canny_inverso

REM Y así sucesivamente...
```

### Ejemplo 3: Modificar Parámetros

Edita `scripts\train_model.bat` y cambia:

```batch
REM Aumentar steps
set STEPS=1000000

REM Cambiar learning rate
set LEARNING_RATE=0.0001

REM Activar visualización de muestras
set SHOW_SAMPLES=--show-samples
```

Luego ejecuta:
```cmd
scripts\train_model.bat canny
```

---

## 🔧 Solución de Problemas

### Error: "python no se reconoce como comando"

**Solución:** Agrega Python al PATH de Windows
1. Busca "Variables de entorno" en Windows
2. Edita la variable PATH
3. Agrega la ruta de Python (ej: `C:\Python39\`)
4. Reinicia CMD/PowerShell

### Error: "No se encontró el directorio dataset/canny/train"

**Solución:** Primero organiza el dataset
```cmd
python organize.py
```

### Error: Faltan dependencias de Python

**Solución:** Instala los requisitos
```cmd
pip install -r requirements.txt
```

### Los scripts .bat no funcionan

**Solución 1:** Ejecuta desde la raíz del proyecto
```cmd
cd C:\Users\choc-\OneDrive\Documents\Proyectos DMREF\pix-2-pix_template
scripts\train_model.bat canny
```

**Solución 2:** Usa Python directamente
```cmd
python train.py --input-path dataset/canny --ground-truth-path "dataset/1051 Redimensionadas" --dataset-name canny_model
```

### Error con rutas que tienen espacios

**Solución:** Usa comillas dobles
```cmd
cd "C:\Users\choc-\OneDrive\Documents\Proyectos DMREF\pix-2-pix_template"
scripts\train_model.bat canny
```

---

## 💡 Tips para Windows

1. **Usa CMD o PowerShell en modo Administrador** si tienes problemas de permisos

2. **Verifica la instalación de Python:**
   ```cmd
   python --version
   pip --version
   ```

3. **Verifica TensorFlow:**
   ```cmd
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

4. **Usa rutas absolutas si tienes problemas:**
   ```cmd
   cd C:\Users\choc-\OneDrive\Documents\Proyectos DMREF\pix-2-pix_template
   python train.py --input-path "C:\...\dataset\canny"
   ```

5. **Crea accesos directos:**
   - Click derecho en `scripts\train_model.bat` → "Crear acceso directo"
   - Mueve el acceso al escritorio
   - Edita propiedades → "Iniciar en" → ruta del proyecto

---

## 📊 Monitoreo en Windows

### TensorBoard

```cmd
REM En una ventana CMD separada:
cd C:\Users\choc-\OneDrive\Documents\Proyectos DMREF\pix-2-pix_template
tensorboard --logdir logs

REM Abre en navegador: http://localhost:6006
```

### Ver progreso en tiempo real

```cmd
REM En PowerShell:
Get-Content .\logs\training.log -Wait

REM O simplemente ve la ventana donde corre el entrenamiento
```

---

## 🎯 Comandos Rápidos

```cmd
REM Ir al proyecto
cd C:\Users\choc-\OneDrive\Documents\Proyectos DMREF\pix-2-pix_template

REM Organizar dataset
python organize.py

REM Prueba rápida
scripts\quick_test.bat canny

REM Entrenar
scripts\train_model.bat canny

REM Predecir
scripts\predict_model.bat canny_model canny

REM Ver métricas
type resultados\canny_model\metricas\test_metrics_summary.txt

REM Ver imágenes generadas
explorer resultados\canny_model\imagenes_prueba
```

---

## 📚 Más Información

- Ver parámetros disponibles: `python train.py --help`
- Documentación completa: [README.md](README.md)
- Guía de scripts bash: [README.md](README.md) (para Git Bash)
