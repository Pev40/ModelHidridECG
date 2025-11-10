@echo off
REM Script de ejemplo para preprocesar datos antes de entrenar

echo ========================================
echo Preprocesamiento de Datos
echo ========================================
echo.

REM Preprocesar MIT-BIH
echo Preprocesando MIT-BIH...
python preprocess_data.py --dataset mit --mitdb_path ./mit-bih-arrhythmia-database-1.0.0

if %ERRORLEVEL% NEQ 0 (
    echo ERROR al preprocesar MIT-BIH
    pause
    exit /b 1
)

echo.
echo MIT-BIH preprocesado exitosamente!
echo.

REM Preprocesar PTB-XL (opcional)
echo Preprocesando PTB-XL...
python preprocess_data.py --dataset ptb --ptbxl_path ./PTBXL/records500 --ptbxl_metadata ./PTBXL/ptbxl_database.csv

if %ERRORLEVEL% NEQ 0 (
    echo ERROR al preprocesar PTB-XL
    pause
    exit /b 1
)

echo.
echo PTB-XL preprocesado exitosamente!
echo.
echo ========================================
echo Preprocesamiento completado!
echo Ahora puedes ejecutar el entrenamiento
echo ========================================
pause

