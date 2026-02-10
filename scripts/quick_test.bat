@echo off
REM Script para hacer una prueba rapida del entrenamiento en Windows
REM Entrena por pocos pasos para verificar que todo funciona
REM Uso: quick_test.bat [algoritmo]

REM ============================================================================
REM CONFIGURACION
REM ============================================================================

REM Cambiar al directorio raiz del proyecto
cd /d "%~dp0\.."

set ALGORITMO=%1
if "%ALGORITMO%"=="" set ALGORITMO=canny

set NOMBRE_MODELO=%ALGORITMO%_test
set STEPS=100
set GROUND_TRUTH_PATH=dataset/1051 Redimensionadas
set INPUT_PATH=dataset/%ALGORITMO%

REM ============================================================================
REM NO MODIFICAR A PARTIR DE AQUI
REM ============================================================================

echo ============================================================
echo Prueba Rapida de Entrenamiento
echo ============================================================
echo Algoritmo:    %ALGORITMO%
echo Steps:        %STEPS% (solo para prueba)
echo Modelo:       %NOMBRE_MODELO%
echo ============================================================
echo.

REM Verificar directorios
if not exist "%INPUT_PATH%\train" (
    echo ERROR: No se encontro %INPUT_PATH%\train
    exit /b 1
)

REM Ejecutar con visualizacion de muestras
python train.py ^
    --input-path "%INPUT_PATH%" ^
    --ground-truth-path "%GROUND_TRUTH_PATH%" ^
    --dataset-name "%NOMBRE_MODELO%" ^
    --steps %STEPS% ^
    --eval-interval 50 ^
    --save-interval 50 ^
    --show-samples ^
    --show-model-summary

echo.
echo ============================================================
echo Prueba completada
echo Si no hubo errores, puedes proceder con entrenamiento completo
echo ============================================================
echo.
echo Para entrenar el modelo completo, ejecuta:
echo   scripts\train_model.bat %ALGORITMO%
