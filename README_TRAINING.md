# Guía de Entrenamiento para TF_MorphoTransNet

## Estructura del Proyecto

```
NewModel/
├── main.py                 # Modelo TF_MorphoTransNet
├── train.py               # Script principal de entrenamiento
├── trainer.py             # Clase trainer con lógica de entrenamiento
├── dataloader.py          # Carga de datos desde archivos .pt
├── utils.py               # Funciones auxiliares
├── preprocesamiento.py    # Preprocesamiento de datasets MIT-BIH y PTB-XL
├── configs/
│   ├── data_configs.py    # Configuraciones de datasets
│   └── hparams.py         # Hyperparámetros de entrenamiento
└── requirements.txt       # Dependencias
```

## Preparación de Datos

**IMPORTANTE**: Antes de entrenar, debes preprocesar los datos. El script `preprocess_data.py` facilita este proceso.

### Opción 1: Usar el script de preprocesamiento (Recomendado)

#### Preprocesar MIT-BIH

```bash
python preprocess_data.py --dataset mit --mitdb_path ./mit-bih-arrhythmia-database-1.0.0
```

Si el dataset está en el directorio actual, el script intentará encontrarlo automáticamente:

```bash
python preprocess_data.py --dataset mit
```

#### Preprocesar PTB-XL

```bash
python preprocess_data.py --dataset ptb --ptbxl_path ./PTBXL/records500 --ptbxl_metadata ./PTBXL/ptbxl_database.csv
```

O con detección automática:

```bash
python preprocess_data.py --dataset ptb
```

### Opción 2: Preprocesar manualmente con Python

```python
from preprocesamiento import process_mitbih_dataset, save_dataset_splits

# Procesar dataset completo
segments, labels = process_mitbih_dataset(
    mitdb_path='./mit-bih-arrhythmia-database-1.0.0',
    target_fs=250,
    window_size=250,
    save_path='./data/mitbih_all.pt'
)

# Guardar splits (train/val/test)
save_dataset_splits(
    segments, labels, 
    output_dir='./data/mit',
    dataset_name='mitbih',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

## Estructura de Directorios de Datos

Después del preprocesamiento, la estructura debe ser:

```
data/
├── mit/
│   ├── train.pt
│   ├── val.pt
│   └── test.pt
└── ptb/
    ├── train.pt
    ├── val.pt
    └── test.pt
```

## Entrenamiento

### Comando Básico

```bash
python train.py --dataset mit --data_path ./data --device cuda:0
```

### Opciones Completas

```bash
python train.py \
    --dataset mit \
    --data_path ./data \
    --device cuda:0 \
    --experiment_description Exp1 \
    --run_description run1 \
    --seed_id 0 \
    --num_runs 1 \
    --use_morph_loss \
    --morph_loss_weight 0.1
```

### Parámetros

- `--dataset`: Dataset a usar (`mit`, `ptb`, `ptb_multi`)
- `--data_path`: Ruta al directorio con los datos preprocesados
- `--device`: Dispositivo (`cpu` o `cuda:0`, `cuda:1`, etc.)
- `--experiment_description`: Nombre del experimento
- `--run_description`: Nombre de la ejecución
- `--seed_id`: Semilla para reproducibilidad
- `--num_runs`: Número de ejecuciones (actualmente soporta 1)
- `--use_morph_loss`: Activar pérdida auxiliar de morfología
- `--morph_loss_weight`: Peso para la pérdida de morfología (default: 0.1)

## Configuración de Hyperparámetros

Editar `configs/hparams.py`:

```python
self.train_params = {
    'num_epochs': 60,
    'batch_size': 64,  # Ajustar según GPU disponible
    'weight_decay': 1e-4,
    'learning_rate': 1e-3,
    'feature_dim': 16
}
```

## Configuración de Dataset

Editar `configs/data_configs.py` para ajustar:
- Número de clases
- Nombres de clases
- Tamaño de secuencia
- Parámetros del modelo (canales, dropout, etc.)

## Resultados

Los resultados se guardan en `experiments_logs/`:

```
experiments_logs/
└── Exp1/
    └── run1_HH_MM/
        └── _seed_0/
            ├── logs_*.log
            ├── checkpoint_best.pt
            ├── checkpoint_last.pt
            ├── classification_report_validation_best.xlsx
            ├── classification_report_test_best.xlsx
            └── MODEL_BACKUP_FILES/
```

## Características del Entrenamiento

1. **Manejo de Desbalance**: Usa pesos de clases calculados automáticamente
2. **Morphology Loss**: Pérdida auxiliar opcional para enfocar atención en regiones QRS
3. **Gradient Clipping**: Previene gradientes explosivos
4. **Checkpoints**: Guarda mejor modelo basado en F1-score de validación
5. **Métricas**: Accuracy y F1-score macro, reportes de clasificación en Excel
6. **Logging**: Logs detallados de entrenamiento y validación

## Monitoreo

Los logs muestran:
- Pérdidas (Total, Clasificación, Morfología si está activa)
- Accuracy y F1-score en train/val/test
- Mejor época y rendimiento
- Número de parámetros del modelo

## Troubleshooting

### Error: "CUDA out of memory"
- Reducir `batch_size` en `configs/hparams.py`
- Usar `--device cpu` para entrenar en CPU (más lento)

### Error: "File not found: train.pt"
- Verificar que los datos estén preprocesados
- Verificar la ruta en `--data_path`
- Asegurar que existan `train.pt`, `val.pt`, `test.pt` en el subdirectorio del dataset

### Error: "Dimension mismatch"
- Verificar que `window_size` en preprocesamiento coincida con `sequence_len` en `data_configs.py`
- Verificar que `feature_dim` en `hparams.py` coincida con `out_channels` del modelo (16)

## Ejemplo Completo

```bash
# 1. Preprocesar datos
python -c "from preprocesamiento import process_mitbih_dataset, save_dataset_splits; segments, labels = process_mitbih_dataset('./mit-bih-arrhythmia-database-1.0.0', save_path='./data/mitbih_all.pt'); save_dataset_splits(segments, labels, './data/mit')"

# 2. Entrenar
python train.py --dataset mit --data_path ./data --device cuda:0 --use_morph_loss

# 3. Evaluar resultados
# Los resultados están en experiments_logs/Exp1/run1_*/_seed_0/
```

