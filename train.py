import os
import argparse
import warnings
from trainer import trainer
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

parser = argparse.ArgumentParser(description='Entrenamiento de TF_MorphoTransNet para clasificación de arritmias ECG')

# ========  Nombre del Experimento ================
parser.add_argument('--save_dir', default='experiments_logs', type=str, 
                    help='Directorio que contiene todos los experimentos')
parser.add_argument('--experiment_description', default='Exp1', type=str, 
                    help='Nombre del experimento')
parser.add_argument('--run_description', default='run1', type=str, 
                    help='Nombre de la ejecución')

# ========= Seleccionar el DATASET ==============
parser.add_argument('--dataset', default='mit', type=str, 
                    help='mit, ptb, ptb_multi')
parser.add_argument('--seed_id', default='0', type=str, 
                    help='Semilla para fijar durante el entrenamiento')

# ========= Configuración del experimento ================
parser.add_argument('--data_path', default='data', type=str, 
                    help='Ruta que contiene el dataset (debe tener subdirectorio con train.pt, val.pt, test.pt)')
parser.add_argument('--num_runs', default=1, type=int, 
                    help='Número de ejecuciones consecutivas con diferentes semillas')
parser.add_argument('--device', default='cuda:0', type=str, 
                    help='cpu o cuda (ej: cuda:0)')

# ========= Parámetros de Morphology Loss ================
parser.add_argument('--use_morph_loss', action='store_true', 
                    help='Usar pérdida auxiliar de morfología (morphology loss)')
parser.add_argument('--morph_loss_weight', default=0.1, type=float, 
                    help='Peso para la pérdida de morfología (default: 0.1)')


args = parser.parse_args()

if __name__ == "__main__":
    print("=" * 50)
    print("Entrenamiento de TF_MorphoTransNet")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {args.data_path}")
    print(f"Device: {args.device}")
    print(f"Use morphology loss: {args.use_morph_loss}")
    if args.use_morph_loss:
        print(f"Morphology loss weight: {args.morph_loss_weight}")
    print("=" * 50)
    
    trainer_instance = trainer(args)
    trainer_instance.train()

