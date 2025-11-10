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
            'batch_size': 64,  # Reducido para modelos más grandes, ajustar según GPU
            'weight_decay': 1e-4,
            'learning_rate': 1e-3,
            'feature_dim': 16  # Coincide con out_channels del modelo (16 * SEBasicBlock.expansion = 16)
        }

