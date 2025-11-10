"""
Script para preprocesar datasets MIT-BIH y PTB-XL.
Genera archivos train.pt, val.pt, test.pt listos para entrenamiento.
"""
import argparse
import os
import sys
from preprocesamiento import (
    process_mitbih_dataset, 
    process_ptbxl_dataset, 
    save_dataset_splits
)


def preprocess_mitbih(mitdb_path, output_dir, target_fs=250, window_size=250, 
                      denoise=True, lead_idx=0, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Preprocesa dataset MIT-BIH."""
    print("=" * 60)
    print("Preprocesando dataset MIT-BIH")
    print("=" * 60)
    print(f"Ruta MIT-BIH: {mitdb_path}")
    print(f"Directorio de salida: {output_dir}")
    print(f"Frecuencia objetivo: {target_fs} Hz")
    print(f"Tamaño de ventana: {window_size}")
    print("=" * 60)
    
    if not os.path.exists(mitdb_path):
        print(f"ERROR: No se encuentra el directorio: {mitdb_path}")
        print("Por favor, verifica la ruta al dataset MIT-BIH")
        return False
    
    # Procesar dataset
    segments, labels = process_mitbih_dataset(
        mitdb_path=mitdb_path,
        target_fs=target_fs,
        window_size=window_size,
        denoise=denoise,
        lead_idx=lead_idx,
        save_path=None  # No guardar archivo completo, solo splits
    )
    
    if len(segments) == 0:
        print("ERROR: No se encontraron segmentos válidos")
        return False
    
    # Guardar splits
    save_dataset_splits(
        segments, labels,
        output_dir=output_dir,
        dataset_name='mitbih',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    print("=" * 60)
    print("¡Preprocesamiento completado exitosamente!")
    print("=" * 60)
    return True


def preprocess_ptbxl(ptbxl_path, output_dir, metadata_path=None, target_fs=250, 
                    window_size=250, denoise=True, lead_idx=1, 
                    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, records_file=None):
    """Preprocesa dataset PTB-XL."""
    print("=" * 60)
    print("Preprocesando dataset PTB-XL")
    print("=" * 60)
    print(f"Ruta PTB-XL: {ptbxl_path}")
    print(f"Directorio de salida: {output_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Archivo RECORDS: {records_file}")
    print(f"Frecuencia objetivo: {target_fs} Hz")
    print(f"Tamaño de ventana: {window_size}")
    print("=" * 60)
    
    if not os.path.exists(ptbxl_path):
        print(f"ERROR: No se encuentra el directorio: {ptbxl_path}")
        print("Por favor, verifica la ruta al dataset PTB-XL")
        return False
    
    # Procesar dataset
    segments, labels = process_ptbxl_dataset(
        ptbxl_path=ptbxl_path,
        metadata_path=metadata_path,
        target_fs=target_fs,
        window_size=window_size,
        denoise=denoise,
        lead_idx=lead_idx,
        save_path=None,  # No guardar archivo completo, solo splits
        records_file=records_file
    )
    
    if len(segments) == 0:
        print("ERROR: No se encontraron segmentos válidos")
        return False
    
    # Guardar splits
    save_dataset_splits(
        segments, labels,
        output_dir=output_dir,
        dataset_name='ptbxl',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    print("=" * 60)
    print("¡Preprocesamiento completado exitosamente!")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description='Preprocesar datasets MIT-BIH y PTB-XL')
    
    parser.add_argument('--dataset', type=str, required=True, choices=['mit', 'ptb'],
                       help='Dataset a preprocesar: mit o ptb')
    
    parser.add_argument('--mitdb_path', type=str, default=None,
                       help='Ruta al directorio con archivos MIT-BIH (ej: ./mit-bih-arrhythmia-database-1.0.0)')
    
    parser.add_argument('--ptbxl_path', type=str, default=None,
                       help='Ruta al directorio records500 de PTB-XL (ej: ./PTBXL/records500)')
    
    parser.add_argument('--ptbxl_metadata', type=str, default=None,
                       help='Ruta al archivo CSV de metadata de PTB-XL (ej: ./PTBXL/ptbxl_database.csv)')
    
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Directorio de salida para los datos preprocesados (default: ./data)')
    
    parser.add_argument('--target_fs', type=int, default=250,
                       help='Frecuencia de muestreo objetivo (default: 250 Hz)')
    
    parser.add_argument('--window_size', type=int, default=250,
                       help='Tamaño de ventana para segmentación (default: 250)')
    
    parser.add_argument('--no_denoise', action='store_true',
                       help='Desactivar denoising (por defecto está activado)')
    
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Proporción para training (default: 0.7)')
    
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Proporción para validación (default: 0.15)')
    
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Proporción para test (default: 0.15)')
    
    args = parser.parse_args()
    
    # Validar ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("ERROR: Los ratios train_ratio + val_ratio + test_ratio deben sumar 1.0")
        sys.exit(1)
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Preprocesar según el dataset
    if args.dataset == 'mit':
        if args.mitdb_path is None:
            # Intentar ruta por defecto
            default_paths = [
                './mit-bih-arrhythmia-database-1.0.0',
                '../mit-bih-arrhythmia-database-1.0.0',
                './NewModel/mit-bih-arrhythmia-database-1.0.0'
            ]
            for path in default_paths:
                if os.path.exists(path):
                    args.mitdb_path = path
                    print(f"Usando ruta por defecto encontrada: {path}")
                    break
            
            if args.mitdb_path is None:
                print("ERROR: No se especificó --mitdb_path y no se encontró una ruta por defecto")
                print("Por favor, especifica la ruta con --mitdb_path")
                sys.exit(1)
        
        output_subdir = os.path.join(args.output_dir, 'mit')
        success = preprocess_mitbih(
            mitdb_path=args.mitdb_path,
            output_dir=output_subdir,
            target_fs=args.target_fs,
            window_size=args.window_size,
            denoise=not args.no_denoise,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
    elif args.dataset == 'ptb':
        if args.ptbxl_path is None:
            # Intentar ruta por defecto
            default_paths = [
                './PTBXL/records500',
                '../PTBXL/records500',
                './NewModel/PTBXL/records500'
            ]
            for path in default_paths:
                if os.path.exists(path):
                    args.ptbxl_path = path
                    print(f"Usando ruta por defecto encontrada: {path}")
                    break
            
            if args.ptbxl_path is None:
                print("ERROR: No se especificó --ptbxl_path y no se encontró una ruta por defecto")
                print("Por favor, especifica la ruta con --ptbxl_path")
                sys.exit(1)
        
        if args.ptbxl_metadata is None:
            # Intentar ruta por defecto para metadata
            default_metadata = [
                './PTBXL/ptbxl_database.csv',
                '../PTBXL/ptbxl_database.csv',
                './NewModel/PTBXL/ptbxl_database.csv'
            ]
            for path in default_metadata:
                if os.path.exists(path):
                    args.ptbxl_metadata = path
                    print(f"Usando metadata por defecto encontrada: {path}")
                    break
        
        # Buscar archivo RECORDS
        records_file = None
        if args.ptbxl_path:
            parent_dir = os.path.dirname(args.ptbxl_path) if os.path.dirname(args.ptbxl_path) else '.'
            potential_records = [
                os.path.join(parent_dir, 'RECORDS'),
                os.path.join(args.ptbxl_path, '..', 'RECORDS'),
                './PTBXL/RECORDS',
                '../PTBXL/RECORDS'
            ]
            for rec_file in potential_records:
                if os.path.exists(rec_file):
                    records_file = rec_file
                    break
        
        output_subdir = os.path.join(args.output_dir, 'ptb')
        success = preprocess_ptbxl(
            ptbxl_path=args.ptbxl_path,
            output_dir=output_subdir,
            metadata_path=args.ptbxl_metadata,
            target_fs=args.target_fs,
            window_size=args.window_size,
            denoise=not args.no_denoise,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            records_file=records_file
        )
    
    if not success:
        sys.exit(1)
    
    print(f"\nLos datos preprocesados están listos en: {output_subdir}")
    print("Ahora puedes ejecutar el entrenamiento con:")
    print(f"  python train.py --dataset {args.dataset} --data_path {args.output_dir}")


if __name__ == "__main__":
    main()

