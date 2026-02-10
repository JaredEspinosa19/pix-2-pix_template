@echo off
REM Script para entrenar modelo Pix2Pix en Windows
REM Uso: train_model.bat [algoritmo] [nombre_modelo] [steps]

REM ============================================================================
REM CONFIGURACION - Modifica estos parametros segun tus necesidades
REM ============================================================================

REM Cambiar al directorio raiz del proyecto
cd /d "%~dp0\.."

REM Algoritmo a usar (canny, laplaciano, canny_inverso, etc.)
set ALGORITMO=%1
if "%ALGORITMO%"=="" set ALGORITMO=canny

REM Nombre del modelo (si no se especifica, usa el nombre del algoritmo)
set NOMBRE_MODELO=%2
if "%NOMBRE_MODELO%"=="" set NOMBRE_MODELO=%ALGORITMO%_model

REM Numero de pasos de entrenamiento
set STEPS=%3
if "%STEPS%"=="" set STEPS=500000

REM Ruta al dataset de entrada (algoritmo)
set INPUT_PATH=dataset/%ALGORITMO%

REM Ruta al ground truth
set GROUND_TRUTH_PATH=dataset/1051 Redimensionadas

REM Dimensiones de las imagenes
set IMG_WIDTH=1024
set IMG_HEIGHT=413

REM Parametros de entrenamiento
set BATCH_SIZE=1
set LEARNING_RATE=0.0002
set LAMBDA_L1=100

REM Intervalos de evaluacion y guardado
set EVAL_INTERVAL=1000
set SAVE_INTERVAL=5000

REM Directorios de checkpoints y logs
set CHECKPOINT_DIR=./training_checkpoints
set LOG_DIR=./logs

REM Opciones adicionales (descomenta las que quieras usar)
REM set SHOW_SAMPLES=--show-samples
REM set SHOW_MODEL_SUMMARY=--show-model-summary
REM set RESTORE_CHECKPOINT=--restore-checkpoint
REM set NO_AUGMENTATION=--no-augmentation

REM ============================================================================
REM NO MODIFICAR A PARTIR DE AQUI
REM ============================================================================

echo ============================================================
echo Entrenamiento Pix2Pix
echo ============================================================
echo Algoritmo:       %ALGORITMO%
echo Modelo:          %NOMBRE_MODELO%
echo Steps:           %STEPS%
echo Input path:      %INPUT_PATH%
echo Ground truth:    %GROUND_TRUTH_PATH%
echo ============================================================
echo.

REM Verificar que exista el directorio de entrada
if not exist "%INPUT_PATH%\train" (
    echo ERROR: No se encontro el directorio %INPUT_PATH%\train
    echo Por favor, ejecuta primero: python organize.py
    exit /b 1
)

REM Verificar que exista el directorio de ground truth
if not exist "%GROUND_TRUTH_PATH%" (
    echo ERROR: No se encontro el directorio %GROUND_TRUTH_PATH%
    exit /b 1
)

REM Ejecutar entrenamiento
python train.py ^
    --input-path "%INPUT_PATH%" ^
    --ground-truth-path "%GROUND_TRUTH_PATH%" ^
    --dataset-name "%NOMBRE_MODELO%" ^
    --steps %STEPS% ^
    --img-width %IMG_WIDTH% ^
    --img-height %IMG_HEIGHT% ^
    --batch-size %BATCH_SIZE% ^
    --learning-rate %LEARNING_RATE% ^
    --lambda-l1 %LAMBDA_L1% ^
    --eval-interval %EVAL_INTERVAL% ^
    --save-interval %SAVE_INTERVAL% ^
    --checkpoint-dir "%CHECKPOINT_DIR%" ^
    --log-dir "%LOG_DIR%" ^
    %SHOW_SAMPLES% ^
    %SHOW_MODEL_SUMMARY% ^
    %RESTORE_CHECKPOINT% ^
    %NO_AUGMENTATION%

echo.
echo ============================================================
echo Entrenamiento completado
echo Resultados guardados en: resultados/%NOMBRE_MODELO%/
echo ============================================================
