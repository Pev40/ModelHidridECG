import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np


class Load_Dataset(Dataset):
    """Dataset loader para archivos .pt con formato {'samples': array, 'labels': array}"""
    
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()

        # Cargar muestras
        x_data = dataset["samples"]

        # Convertir a tensor torch
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)
        
        # Asegurar que tenga la forma correcta: (n_samples, 1, sequence_len)
        if len(x_data.shape) == 2:
            # Agregar dimensión de canal si falta
            x_data = x_data.unsqueeze(1)

        # Cargar etiquetas
        y_data = dataset.get("labels")
        if y_data is not None:
            if isinstance(y_data, np.ndarray):
                y_data = torch.from_numpy(y_data)

        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None

        self.len = x_data.shape[0]

    def get_labels(self):
        """Retorna todas las etiquetas"""
        return self.y_data

    def __getitem__(self, index):
        sample = {
            'samples': self.x_data[index],
            'labels': int(self.y_data[index])
        }
        return sample

    def __len__(self):
        return self.len


def data_generator(data_path, data_type, hparams, configs=None):
    """
    Genera dataloaders para train, validation y test.
    
    Args:
        data_path: Ruta base donde están los datos
        data_type: Tipo de dataset (mit, ptb, etc.)
        hparams: Hyperparámetros con batch_size
        configs: Configuración del dataset (si no se provee, se importa)
        
    Returns:
        train_loader, val_loader, test_loader, class_weights_dict
    """
    # Importar configs si no se provee
    if configs is None:
        from configs.data_configs import get_dataset_class
        configs = get_dataset_class(data_type)()
    # Verificar que los archivos existan
    train_path = os.path.join(data_path, data_type, f"train.pt")
    val_path = os.path.join(data_path, data_type, f"val.pt")
    test_path = os.path.join(data_path, data_type, f"test.pt")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Archivo no encontrado: {train_path}\n"
            f"Por favor, preprocesa los datos primero usando preprocesamiento.py\n"
            f"Ejemplo:\n"
            f"  python preprocess_data.py --dataset {data_type}"
        )
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Archivo no encontrado: {val_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Archivo no encontrado: {test_path}")
    
    # Cargar datasets desde archivos .pt
    # weights_only=False necesario porque los archivos contienen arrays de numpy
    train_dataset = torch.load(train_path, weights_only=False)
    val_dataset = torch.load(val_path, weights_only=False)
    test_dataset = torch.load(test_path, weights_only=False)

    # Crear datasets
    train_dataset = Load_Dataset(train_dataset)
    val_dataset = Load_Dataset(val_dataset)
    test_dataset = Load_Dataset(test_dataset)

    # Remapear etiquetas si hay más clases en los datos que las esperadas por el modelo
    unique_classes = np.unique(train_dataset.y_data.numpy())
    num_classes_in_data = len(unique_classes)
    num_classes_expected = configs.num_classes
    
    if num_classes_in_data > num_classes_expected:
        # Necesitamos remapear las clases a las esperadas
        # Para PTB binario: mapear todas las clases anormales a 1
        if num_classes_expected == 2:
            # Mapear: clase 0 (normal) -> 0, todas las demás -> 1 (anormal)
            print(f"Remapeando {num_classes_in_data} clases a 2 clases (binario):")
            print(f"  Clases encontradas: {sorted(unique_classes)}")
            print(f"  Mapeo: 0 -> 0 (normal), {sorted(set(unique_classes) - {0})} -> 1 (anormal)")
            
            # Remapear en todos los datasets
            for dataset in [train_dataset, val_dataset, test_dataset]:
                y_data = dataset.y_data.numpy()
                # Crear máscara: 0 -> 0, todo lo demás -> 1
                y_remapped = np.where(y_data == 0, 0, 1)
                dataset.y_data = torch.from_numpy(y_remapped).long()
            
            # Recalcular unique_classes después del remapeo
            unique_classes = np.unique(train_dataset.y_data.numpy())
            print(f"  Después del remapeo: clases {sorted(unique_classes)}")
        else:
            # Para otros casos, tomar solo las primeras num_classes_expected clases
            print(f"Advertencia: Remapeando {num_classes_in_data} clases a {num_classes_expected} clases")
            print(f"  Usando solo las primeras {num_classes_expected} clases: {sorted(unique_classes[:num_classes_expected])}")
            
            for dataset in [train_dataset, val_dataset, test_dataset]:
                y_data = dataset.y_data.numpy()
                # Remapear: mantener las primeras num_classes_expected, el resto a la última clase
                y_remapped = np.where(y_data < num_classes_expected, y_data, num_classes_expected - 1)
                dataset.y_data = torch.from_numpy(y_remapped).long()
            
            unique_classes = np.unique(train_dataset.y_data.numpy())
    
    # Calcular pesos de clases para manejo de desbalance
    # cw_dict contiene el conteo de cada clase en los datos de entrenamiento (después del remapeo)
    cw = train_dataset.y_data.numpy().tolist()
    cw_dict = {}
    for cls in unique_classes:
        cw_dict[int(cls)] = cw.count(int(cls))
    
    print(f"Distribución de clases después del remapeo: {cw_dict}")

    # Crear dataloaders con configuración optimizada
    batch_size = hparams.get("batch_size", 512)
    num_workers = hparams.get("num_workers", 8)
    pin_memory = hparams.get("pin_memory", True) if torch.cuda.is_available() else False
    prefetch_factor = hparams.get("prefetch_factor", 4) if num_workers > 0 else None
    
    # Ajustar num_workers según disponibilidad
    if num_workers > 0:
        import multiprocessing
        max_workers = multiprocessing.cpu_count()
        num_workers = min(num_workers, max_workers)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        drop_last=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        drop_last=True,  # Mantener consistente con train para mejor uso de GPU
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        drop_last=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader, cw_dict

