import torch
import torch.nn.functional as F
import datetime
import os
import collections
import numpy as np

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from main import TF_MorphoTransNet
from dataloader import data_generator
from configs.data_configs import get_dataset_class
from configs.hparams import get_hparams_class
from utils import AverageMeter, to_device, _save_metrics, copy_Files
from utils import fix_randomness, starting_logs, save_checkpoint, _calc_metrics, get_class_weight, count_parameters


class trainer(object):
    def __init__(self, args):
        # Parámetros del dataset
        self.dataset = args.dataset
        self.seed_id = args.seed_id

        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # Descripción del experimento
        self.run_description = f"{args.run_description}_{datetime.datetime.now().strftime('%H_%M')}"
        self.experiment_description = args.experiment_description

        # Rutas
        self.home_path = os.getcwd()
        self.save_dir = os.path.join(os.getcwd(), "experiments_logs")
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, self.run_description)
        os.makedirs(self.exp_log_dir, exist_ok=True)

        self.data_path = args.data_path

        # Número de ejecuciones
        self.num_runs = args.num_runs

        # Obtener configuraciones del dataset y hyperparámetros
        self.dataset_configs, self.hparams_class = self.get_configs()

        # Hyperparámetros
        self.hparams = self.hparams_class.train_params
        
        # Parámetros adicionales
        self.use_morph_loss = getattr(args, 'use_morph_loss', False)  # Si usar morphology loss
        self.morph_loss_weight = getattr(args, 'morph_loss_weight', 0.1)  # Peso para morphology loss

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class("supervised")
        return dataset_class(), hparams_class()

    def load_data(self, data_type):
        self.train_dl, self.val_dl, self.test_dl, self.cw_dict = \
            data_generator(self.data_path, data_type, self.hparams)

    def calc_results_per_run(self):
        acc, f1 = _calc_metrics(self.pred_labels, self.true_labels, self.dataset_configs.class_names)
        return acc, f1

    def train(self):
        # Copiar archivos del modelo
        source_files = [
            "main.py",
            "dataloader.py",
            "configs/data_configs.py",
            "configs/hparams.py",
            "trainer.py",
            "utils.py",
            "preprocesamiento.py"
        ]
        copy_Files(self.exp_log_dir, source_files)

        self.metrics = {'accuracy': [], 'f1_score': []}

        # Fijar semilla aleatoria
        fix_randomness(int(self.seed_id))

        # Logging
        self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.exp_log_dir, self.seed_id)
        self.logger.debug(self.hparams)
        self.logger.debug(f"Usando morphology loss: {self.use_morph_loss}")
        if self.use_morph_loss:
            self.logger.debug(f"Peso de morphology loss: {self.morph_loss_weight}")
        
        # Cargar datos
        self.load_data(self.dataset)

        # Crear modelo
        model = TF_MorphoTransNet(configs=self.dataset_configs, hparams=self.hparams)
        model.to(self.device)
        
        # Contar parámetros
        num_params = count_parameters(model)
        self.logger.debug(f"Número de parámetros del modelo: {num_params:,}")

        # Average meters
        loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

        # Optimizador
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
            betas=(0.9, 0.99)
        )
        
        # Calcular pesos de clases para manejo de desbalance
        class_weights = get_class_weight(self.cw_dict)
        weights = [class_weights.get(i, 1.0) for i in range(len(self.cw_dict))]
        weights_array = np.array(weights).astype(np.float32)
        weights_tensor = torch.tensor(weights_array).to(self.device)
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=weights_tensor)
        
        self.logger.debug(f"Pesos de clases: {dict(zip(range(len(weights)), weights))}")

        best_acc = 0
        best_f1 = 0
        best_epoch = 0

        # Entrenamiento
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            model.train()

            for step, batches in enumerate(self.train_dl):
                batches = to_device(batches, self.device)

                data = batches['samples'].float()
                labels = batches['labels'].long()

                # Forward pass
                self.optimizer.zero_grad()

                # El modelo retorna (logits, morph_loss)
                if self.use_morph_loss:
                    logits, morph_loss = model(data, use_attn=True, compute_morph_loss=True)
                else:
                    logits, morph_loss = model(data, use_attn=False, compute_morph_loss=False)

                # Pérdida de clasificación
                cls_loss = self.cross_entropy(logits, labels)

                # Pérdida total
                if self.use_morph_loss and morph_loss is not None:
                    total_loss = cls_loss + self.morph_loss_weight * morph_loss
                    loss_avg_meters['Morph_loss'].update(morph_loss.item(), self.hparams["batch_size"])
                else:
                    total_loss = cls_loss
                    morph_loss = None

                # Backward pass
                total_loss.backward()
                
                # Gradient clipping para estabilidad
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                self.optimizer.step()

                # Actualizar métricas
                loss_avg_meters['Total_loss'].update(total_loss.item(), self.hparams["batch_size"])
                loss_avg_meters['Cls_loss'].update(cls_loss.item(), self.hparams["batch_size"])

            # Logging de pérdidas
            self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in loss_avg_meters.items():
                self.logger.debug(f'{key}\t: {val.avg:2.4f}')

            # Validación
            self.evaluate(model, self.val_dl, use_morph_loss=False)  # No necesitamos morph_loss en validación
            val_acc, val_f1 = self.calc_results_per_run()
            
            # Guardar mejor modelo basado en F1-score
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                best_epoch = epoch
                save_checkpoint(self.exp_log_dir, model, self.dataset, self.dataset_configs, self.hparams, "best")
                _save_metrics(self.pred_labels, self.true_labels, self.scenario_log_dir, "validation_best")
                self.logger.debug(f"*** NUEVO MEJOR MODELO *** (Epoch {epoch})")

            # Logging
            self.logger.debug(f'VAL  : Acc:{val_acc:2.4f} \t F1:{val_f1:2.4f} (best: {best_f1:2.4f} @ epoch {best_epoch})')
            self.logger.debug(f'-------------------------------------')

        # Última época
        _save_metrics(self.pred_labels, self.true_labels, self.scenario_log_dir, "validation_last")
        self.logger.debug("RENDIMIENTO DE ÚLTIMA ÉPOCA en conjunto de validación...")
        self.logger.debug(f'Acc:{val_acc:2.4f} \t F1:{val_f1:2.4f}')

        self.logger.debug(":::::::::::::")
        # Mejor época
        self.logger.debug("RENDIMIENTO DE MEJOR ÉPOCA en conjunto de validación...")
        self.logger.debug(f'Acc:{best_acc:2.4f} \t F1:{best_f1:2.4f} (Epoch {best_epoch})')
        save_checkpoint(self.exp_log_dir, model, self.dataset, self.dataset_configs, self.hparams, "last")

        # Testing
        self.logger.debug(" === Evaluando en conjunto de TEST ===")
        self.evaluate(model, self.test_dl, use_morph_loss=False)
        test_acc, test_f1 = self.calc_results_per_run()
        _save_metrics(self.pred_labels, self.true_labels, self.scenario_log_dir, "test_last")
        self.logger.debug(f'TEST: Acc:{test_acc:2.4f} \t F1:{test_f1:2.4f}')
        
        # Cargar mejor modelo y evaluar en test
        self.logger.debug(" === Evaluando MEJOR modelo en conjunto de TEST ===")
        checkpoint_path = os.path.join(self.exp_log_dir, "checkpoint_best.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            model.to(self.device)
            self.evaluate(model, self.test_dl, use_morph_loss=False)
            test_acc_best, test_f1_best = self.calc_results_per_run()
            _save_metrics(self.pred_labels, self.true_labels, self.scenario_log_dir, "test_best")
            self.logger.debug(f'TEST (best model): Acc:{test_acc_best:2.4f} \t F1:{test_f1_best:2.4f}')

    def evaluate(self, model, dataset, use_morph_loss=False):
        """Evalúa el modelo en un conjunto de datos."""
        model.to(self.device).eval()

        total_loss_ = []
        self.pred_labels = np.array([])
        self.true_labels = np.array([])

        with torch.no_grad():
            for batches in dataset:
                batches = to_device(batches, self.device)
                data = batches['samples'].float()
                labels = batches['labels'].long()

                # Forward pass
                logits, morph_loss = model(data, use_attn=use_morph_loss, compute_morph_loss=use_morph_loss)

                # Calcular pérdida
                loss = self.cross_entropy(logits, labels)
                if use_morph_loss and morph_loss is not None:
                    loss = loss + self.morph_loss_weight * morph_loss
                total_loss_.append(loss.item())
                
                # Obtener predicciones
                pred = logits.detach().argmax(dim=1)

                self.pred_labels = np.append(self.pred_labels, pred.cpu().numpy())
                self.true_labels = np.append(self.true_labels, labels.data.cpu().numpy())

        self.trg_loss = torch.tensor(total_loss_).mean()  # pérdida promedio

