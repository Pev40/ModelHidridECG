@echo off
REM Script de ejemplo para entrenar el modelo en Windows

REM Ejemplo 1: Entrenar en MIT-BIH sin morphology loss
python train.py --dataset mit --data_path ./data --device cuda:0 --experiment_description MIT_Exp1 --run_description baseline --seed_id 0

REM Ejemplo 2: Entrenar en MIT-BIH con morphology loss
python train.py --dataset mit --data_path ./data --device cuda:0 --experiment_description MIT_Exp2 --run_description with_morph_loss --seed_id 0 --use_morph_loss --morph_loss_weight 0.1

REM Ejemplo 3: Entrenar en PTB-XL
python train.py --dataset ptb --data_path ./data --device cuda:0 --experiment_description PTB_Exp1 --run_description baseline --seed_id 0

