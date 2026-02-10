@echo off
REM Script para realizar predicciones con modelo Pix2Pix entrenado en Windows
REM Uso: predict_model.bat [nombre_modelo] [algoritmo]

REM ============================================================================
REM CONFIGURACION - Modifica estos parametros segun tus necesidades
REM ============================================================================

REM Cambiar al directorio raiz del proyecto
cd /d "%~dp0\.."

REM Nombre del modelo entrenado
set NOMBRE_MODELO=%1
if "%NOMBRE_MODELO%"=="" set NOMBRE_MODELO=canny_model

REM Algoritmo usado (para encontrar el test dataset)
set ALGORITMO=%2
if "%ALGORITMO%"=="" set ALGORITMO=canny

REM Ruta a los pesos del modelo
set WEIGHTS_PATH=resultados/%NOMBRE_MODELO%/weights/best_generator_weights.h5

REM Ruta al dataset de prueba
set TEST_DATASET=dataset/%ALGORITMO%

REM Ruta al ground truth
set GROUND_TRUTH_PATH=dataset/1051 Redimensionadas

REM Directorio de salida (usa el mismo del modelo)
set OUTPUT_DIR=resultados/%NOMBRE_MODELO%

REM Dimensiones de las imagenes
set IMG_WIDTH=1024
set IMG_HEIGHT=413

REM Opciones adicionales (descomenta la que quieras usar)
REM set VISUALIZE=--visualize

REM ============================================================================
REM NO MODIFICAR A PARTIR DE AQUI
REM ============================================================================

echo ============================================================
echo Prediccion Pix2Pix
echo ============================================================
echo Modelo:          %NOMBRE_MODELO%
echo Algoritmo:       %ALGORITMO%
echo Weights:         %WEIGHTS_PATH%
echo Test dataset:    %TEST_DATASET%
echo Output dir:      %OUTPUT_DIR%
echo ============================================================
echo.

REM Verificar que existan los pesos
if not exist "%WEIGHTS_PATH%" (
    echo ERROR: No se encontro el archivo de pesos: %WEIGHTS_PATH%
    echo Por favor, verifica que el modelo este entrenado
    exit /b 1
)

REM Verificar que exista el directorio de test
if not exist "%TEST_DATASET%\test" (
    echo ERROR: No se encontro el directorio %TEST_DATASET%\test
    exit /b 1
)

REM Ejecutar prediccion
python predict.py ^
    --weights "%WEIGHTS_PATH%" ^
    --test-dataset "%TEST_DATASET%" ^
    --ground-truth-path "%GROUND_TRUTH_PATH%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --img-width %IMG_WIDTH% ^
    --img-height %IMG_HEIGHT% ^
    %VISUALIZE%

echo.
echo ============================================================
echo Prediccion completada
echo Imagenes guardadas en: %OUTPUT_DIR%/imagenes_prueba/
echo Metricas guardadas en: %OUTPUT_DIR%/metricas/
echo ============================================================
