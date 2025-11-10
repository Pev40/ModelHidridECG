import torch
import random
import os
import sys
import logging
import numpy as np
import pandas as pd
from shutil import copy
from datetime import datetime
import matplotlib.pyplot as plt
import collections

from sklearn.metrics import classification_report, accuracy_score, f1_score


def count_parameters(model):
    """Cuenta el número de parámetros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computa y almacena el promedio y valor actual"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fix_randomness(SEED):
    """Fija la semilla para reproducibilidad."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _logger(logger_name, level=logging.DEBUG):
    """
    Método para retornar un logger personalizado con el nombre y nivel dados
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Crear y agregar el console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Crear y agregar el file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def starting_logs(data_type, exp_log_dir, seed_id):
    """Inicializa el sistema de logging."""
    log_dir = os.path.join(exp_log_dir, "_seed_" + str(seed_id))
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug("=" * 45)
    logger.debug(f'Seed: {seed_id}')
    logger.debug("=" * 45)
    return logger, log_dir


def save_checkpoint(exp_log_dir, model, dataset, dataset_configs, hparams, status):
    """Guarda un checkpoint del modelo."""
    save_dict = {
        "dataset": dataset,
        "configs": dataset_configs.__dict__,
        "hparams": dict(hparams),
        "model": model.state_dict()
    }
    # Guardar checkpoint
    save_path = os.path.join(exp_log_dir, f"checkpoint_{status}.pt")
    torch.save(save_dict, save_path)
    print(f"Checkpoint guardado en: {save_path}")


def _calc_metrics(pred_labels, true_labels, classes_names):
    """Calcula métricas de clasificación (accuracy y F1-score macro)."""
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # Calcular accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # Calcular F1-score macro
    try:
        f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    except:
        f1_macro = 0.0
    
    # Calcular classification report completo
    try:
        r = classification_report(true_labels, pred_labels, target_names=classes_names, 
                                  digits=6, output_dict=True, zero_division=0)
    except:
        r = {}
        r["macro avg"] = {"f1-score": f1_macro}

    return accuracy * 100, r["macro avg"]["f1-score"] * 100


def _save_metrics(pred_labels, true_labels, log_dir, status):
    """Guarda métricas de clasificación en archivo Excel."""
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    try:
        r = classification_report(true_labels, pred_labels, digits=6, output_dict=True, zero_division=0)
        df = pd.DataFrame(r)
        accuracy = accuracy_score(true_labels, pred_labels)
        df["accuracy"] = accuracy
        df = df * 100

        # Guardar classification report
        file_name = f"classification_report_{status}.xlsx"
        report_save_path = os.path.join(log_dir, file_name)
        df.to_excel(report_save_path)
        print(f"Métricas guardadas en: {report_save_path}")
    except Exception as e:
        print(f"Error al guardar métricas: {e}")


def to_device(input, device):
    """Mueve input (tensor, dict, list) al dispositivo especificado."""
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.abc.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.abc.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")


def copy_Files(destination, source_files):
    """Copia archivos del modelo al directorio de destino."""
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    
    for file_path in source_files:
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            copy(file_path, os.path.join(destination_dir, file_name))
        else:
            print(f"Advertencia: Archivo no encontrado: {file_path}")


def get_class_weight(labels_dict):
    """
    Calcula pesos de clases para manejar desbalance.
    Usa fórmula logarítmica para suavizar los pesos.
    """
    total = sum(labels_dict.values())
    if total == 0:
        return {}
    
    max_num = max(labels_dict.values())
    mu = 1.0 / (total / max_num)
    class_weight = dict()
    for key, value in labels_dict.items():
        score = np.log(mu * total / float(value)) if value > 0 else 0.0
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight

