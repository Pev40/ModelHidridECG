def get_hparams_class(dataset_name):
    """Retorna la clase de hyperparámetros con el nombre dado."""
    if dataset_name not in globals():
        raise NotImplementedError("Hyperparámetros no encontrados: {}".format(dataset_name))
    return globals()[dataset_name]


class supervised():
    def __init__(self):
        super(supervised, self).__init__()
        self.train_params = {
            'num_epochs': 60,
            'batch_size': 512,  # Aumentado para GPU de alta capacidad (RTX PRO 6000 tiene ~98GB VRAM)
            'weight_decay': 1e-4,
            'learning_rate': 1e-3,
            'feature_dim': 16,  # Coincide con out_channels del modelo (16 * SEBasicBlock.expansion = 16)
            'num_workers': 8,  # Número de workers para DataLoader
            'pin_memory': True,  # Pin memory para transferencia más rápida a GPU
            'prefetch_factor': 4,  # Prefetch batches
            'use_mixed_precision': True,  # Usar FP16 mixed precision
            'use_compile': True,  # Usar torch.compile si está disponible
        }

