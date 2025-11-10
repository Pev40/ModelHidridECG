# Guía de Optimización GPU para TF_MorphoTransNet

## Optimizaciones Implementadas

### 1. Mixed Precision Training (FP16)
- **Habilitado por defecto**: Reduce uso de memoria y acelera entrenamiento
- **GradScaler**: Maneja el escalado de gradientes para estabilidad numérica
- **Aceleración esperada**: ~1.5-2x más rápido, ~50% menos memoria

### 2. Batch Size Optimizado
- **Batch size**: 512 (aumentado desde 64)
- **Para RTX PRO 6000**: Con ~98GB VRAM, puede manejar batches aún más grandes
- **Ajuste dinámico**: Puedes aumentar más si tienes más VRAM disponible

### 3. DataLoader Optimizado
- **num_workers**: 8 workers para carga paralela de datos
- **pin_memory**: True para transferencia más rápida CPU→GPU
- **prefetch_factor**: 4 batches pre-cargados
- **persistent_workers**: True para reutilizar workers entre épocas

### 4. torch.compile (PyTorch 2.0+)
- **Habilitado por defecto**: Compila el modelo para optimización
- **Modo**: 'reduce-overhead' (balance entre compilación y overhead)
- **Aceleración esperada**: ~10-30% más rápido después de warmup

### 5. Optimizador AdamW con Fused Operations
- **AdamW**: Más estable que Adam
- **Fused**: Operaciones fusionadas para GPU (si está disponible)
- **Aceleración**: ~5-10% más rápido

### 6. Configuración cuDNN y TF32
- **benchmark**: True - Optimiza kernels para tamaños fijos
- **deterministic**: False - Permite optimizaciones no determinísticas
- **TF32**: Habilitado usando nueva API (PyTorch 2.0+) para GPUs Ampere+ (acelera FP32)
  - Nueva API: `torch.backends.cuda.matmul.fp32_precision = 'tf32'`
  - Compatible con API antigua si la nueva no está disponible

### 7. Ventana STFT
- **Hann window**: Mejor calidad espectral que rectangular
- **register_buffer**: Ventana registrada como buffer del modelo

## Ajuste de Batch Size

Para maximizar el uso de GPU, ajusta el batch_size en `configs/hparams.py`:

```python
'batch_size': 512,  # Aumentar si tienes más VRAM
```

**Recomendaciones por GPU:**
- RTX PRO 6000 (98GB): 512-1024
- RTX 4090 (24GB): 256-512
- RTX 3090 (24GB): 256-512
- RTX 3080 (10GB): 128-256
- RTX 2080 (8GB): 64-128

## Monitoreo de GPU

Usa `nvidia-smi` para monitorear:
- **GPU-Util**: Debe estar cerca de 100%
- **Memory-Usage**: Debe usar la mayor parte de VRAM disponible
- **Power Usage**: Debe estar cerca del máximo si está trabajando al 100%

## Ajustes Adicionales

### Aumentar Batch Size
Si la GPU no está al 100% de uso:
1. Aumenta `batch_size` en `configs/hparams.py`
2. Si hay OOM (out of memory), reduce ligeramente
3. Monitorea con `nvidia-smi`

### Ajustar Num Workers
Si la CPU es el cuello de botella:
1. Aumenta `num_workers` en `configs/hparams.py`
2. Máximo recomendado: número de cores CPU
3. Demasiados workers pueden ralentizar

### Desactivar Mixed Precision
Si hay problemas de estabilidad numérica:
1. Establece `use_mixed_precision: False` en `configs/hparams.py`
2. Usará más memoria pero será más estable

### Modo de Compilación
En `trainer.py`, puedes cambiar el modo de compilación:
- `'reduce-overhead'`: Balance (recomendado)
- `'max-autotune'`: Máxima optimización (más lento en compilación inicial)
- `'default'`: Compilación rápida

## Troubleshooting

### GPU-Util bajo (<50%)
- Aumenta batch_size
- Aumenta num_workers
- Verifica que pin_memory esté activado
- Verifica que no haya cuello de botella en CPU

### Out of Memory (OOM)
- Reduce batch_size
- Desactiva mixed precision (usa más memoria)
- Reduce num_workers
- Verifica que no haya otros procesos usando GPU

### Velocidad lenta
- Verifica que mixed precision esté activado
- Verifica que torch.compile esté funcionando
- Verifica que cuDNN benchmark esté activado
- Verifica que TF32 esté habilitado (GPUs Ampere+)

## Verificación de Optimizaciones

El código automáticamente muestra en los logs:
- GPU name y VRAM total
- Mixed precision status
- Batch size
- Num workers
- Si torch.compile fue exitoso

