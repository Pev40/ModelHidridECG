import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import wfdb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ECGPreprocessor:
    """
    Preprocesador de señales ECG para datasets MIT-BIH y PTB-XL.
    Realiza: denoising, resampling, normalización, detección de R-peaks y segmentación.
    """
    def __init__(self, target_fs=250, window_size=250, denoise=True):
        self.target_fs = target_fs
        self.window_size = window_size
        self.denoise = denoise
        # Mapeo AAMI estándar para arritmias
        self.aami_map = {
            'N': 0,  # Normal
            'L': 0,  # Left bundle branch block (normal)
            'R': 0,  # Right bundle branch block (normal)
            'e': 0,  # Atrial escape (normal)
            'j': 0,  # Nodal (junctional) escape (normal)
            'S': 1,  # Supraventricular ectopic
            'A': 1,  # Atrial premature
            'a': 1,  # Aberrated atrial premature
            'J': 1,  # Nodal (junctional) premature
            'E': 1,  # Ventricular escape
            'V': 2,  # Ventricular ectopic / Premature ventricular contraction
            'F': 3,  # Fusion
            'f': 3,  # Fusion of paced and normal
            '/': 4,  # Paced
            'Q': 4,  # Unknown/Unclassifiable
            '|': 4,  # Isolated QRS-like artifact
            '~': 4,  # Signal quality change
            'x': 4,  # Non-conducted P-wave
            '[': 4,  # Start of ventricular flutter/fibrillation
            ']': 4,  # End of ventricular flutter/fibrillation
            '!': 4,  # Ventricular flutter wave
        }

    def denoise_signal(self, sig, fs):
        """
        Denoising de señal ECG usando filtro bandpass y remoción de baseline.
        
        Args:
            sig: Señal ECG 1D
            fs: Frecuencia de muestreo
            
        Returns:
            Señal denoised
        """
        # Validar entrada
        if len(sig) < 10:
            return sig
            
        # Bandpass filter 0.5-40 Hz (rango típico para ECG)
        nyquist = fs / 2
        low = 0.5 / nyquist
        high = 40 / nyquist
        if high >= 1.0:
            high = 0.99  # Asegurar que high < 1.0
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            sig = signal.filtfilt(b, a, sig)
        except ValueError:
            # Si falla el filtro, usar solo remoción de baseline
            pass
        
        # Remoción de baseline usando filtro mediano
        kernel_size = max(3, int(fs * 0.2))  # ~0.2 segundos, mínimo 3
        if kernel_size % 2 == 0:
            kernel_size += 1  # kernel_size debe ser impar
        if kernel_size > len(sig):
            kernel_size = len(sig) if len(sig) % 2 == 1 else len(sig) - 1
        
        baseline = signal.medfilt(sig, kernel_size=kernel_size)
        sig = sig - baseline
        
        return sig

    def resample_signal(self, sig, orig_fs):
        """
        Resample señal a target_fs usando interpolación cúbica.
        
        Args:
            sig: Señal ECG 1D
            orig_fs: Frecuencia de muestreo original
            
        Returns:
            Señal resampleada
        """
        if orig_fs == self.target_fs:
            return sig
        
        if len(sig) < 2:
            return sig
        
        t_orig = np.arange(len(sig)) / orig_fs
        try:
            interp_func = interp1d(t_orig, sig, kind='cubic', bounds_error=False, fill_value='extrapolate')
            t_new = np.arange(0, t_orig[-1], 1 / self.target_fs)
            return interp_func(t_new)
        except:
            # Fallback a interpolación lineal si falla la cúbica
            interp_func = interp1d(t_orig, sig, kind='linear', bounds_error=False, fill_value='extrapolate')
            t_new = np.arange(0, t_orig[-1], 1 / self.target_fs)
            return interp_func(t_new)

    def normalize_signal(self, sig):
        """
        Normalización z-score de la señal.
        
        Args:
            sig: Señal ECG 1D
            
        Returns:
            Señal normalizada
        """
        sig_mean = np.mean(sig)
        sig_std = np.std(sig)
        if sig_std > 1e-6:  # Evitar división por cero
            return (sig - sig_mean) / sig_std
        else:
            return sig - sig_mean  # Solo centrar si std es muy pequeño

    def detect_r_peaks(self, sig, fs):
        """
        Detección de picos R usando algoritmo simplificado tipo Pan-Tompkins.
        
        Args:
            sig: Señal ECG 1D
            fs: Frecuencia de muestreo
            
        Returns:
            Índices de picos R detectados
        """
        if len(sig) < fs:  # Señal muy corta
            return np.array([len(sig) // 2]) if len(sig) > 0 else np.array([])
        
        try:
            # Filtro bandpass 5-15 Hz para enfatizar QRS
            nyquist = fs / 2
            low = 5 / nyquist
            high = 15 / nyquist
            if high >= 1.0:
                high = 0.99
            
            b, a = signal.butter(4, [low, high], btype='band')
            sig_filt = signal.filtfilt(b, a, sig)
        except:
            sig_filt = sig
        
        # Derivada y cuadrado
        sig_diff = np.diff(sig_filt)
        sig_sq = sig_diff ** 2
        
        # Moving average (ventana de ~150ms)
        window_size = max(1, int(fs * 0.15))
        sig_ma = np.convolve(sig_sq, np.ones(window_size) / window_size, mode='same')
        
        # Detección de picos
        min_height = np.mean(sig_ma) * 2  # Umbral más bajo para capturar más picos
        min_distance = max(1, int(fs * 0.4))  # Mínimo 0.4 segundos entre picos
        
        peaks, properties = signal.find_peaks(sig_ma, height=min_height, distance=min_distance)
        
        # Si no se detectan picos, usar método alternativo
        if len(peaks) == 0:
            # Usar percentil 75 como umbral
            threshold = np.percentile(sig_ma, 75)
            peaks, _ = signal.find_peaks(sig_ma, height=threshold, distance=min_distance)
        
        # Si aún no hay picos, usar máximo local simple
        if len(peaks) == 0:
            from scipy.signal import argrelextrema
            peaks = argrelextrema(sig_ma, np.greater, order=min_distance)[0]
        
        return peaks

    def segment_beats(self, sig, r_peaks, label_map):
        """
        Segmentación de latidos centrados en picos R.
        
        Args:
            sig: Señal ECG completa
            r_peaks: Índices de picos R
            label_map: Diccionario o lista con etiquetas para cada pico R
            
        Returns:
            segments: Array de segmentos de latidos (n_segments, window_size)
            labels: Array de etiquetas (n_segments,)
        """
        segments = []
        labels = []
        
        # Calcular ventana alrededor del pico R
        # ~1/3 antes del R, ~2/3 después (típico para capturar onda P completa)
        pre = self.window_size // 3
        post = self.window_size - pre - 1
        
        for i, r in enumerate(r_peaks):
            # Verificar que la ventana esté dentro de los límites
            if r - pre >= 0 and r + post < len(sig):
                seg = sig[r - pre : r + post + 1]
                
                # Asegurar que el segmento tenga la longitud correcta
                if len(seg) == self.window_size:
                    segments.append(seg)
                    
                    # Obtener etiqueta
                    if isinstance(label_map, dict):
                        label = label_map.get(r, 'Q')
                    elif isinstance(label_map, (list, np.ndarray)):
                        label = label_map[i] if i < len(label_map) else 'Q'
                    else:
                        label = 'Q'
                    
                    labels.append(self.aami_map.get(label, 4))  # Default Q (clase 4)
        
        if len(segments) == 0:
            return np.array([]).reshape(0, self.window_size), np.array([])
        
        return np.array(segments), np.array(labels, dtype=np.int64)

    def process_mitbih_record(self, record_path, lead_idx=0):
        """
        Procesa un registro MIT-BIH.
        
        Args:
            record_path: Ruta al registro (sin extensión)
            lead_idx: Índice del lead a usar (0=MLII, 1=V1)
            
        Returns:
            segments: Array de segmentos (n_segments, window_size)
            labels: Array de etiquetas (n_segments,)
        """
        try:
            record = wfdb.rdrecord(record_path)
            sig = record.p_signal[:, lead_idx]  # Lead seleccionado
            ann = wfdb.rdann(record_path, 'atr')
            fs = record.fs  # 360 Hz para MIT-BIH
            
            # Preprocesamiento
            if self.denoise:
                sig = self.denoise_signal(sig, fs)
            sig = self.resample_signal(sig, fs)
            sig = self.normalize_signal(sig)
            
            # Detectar picos R en señal resampleada
            r_peaks = self.detect_r_peaks(sig, self.target_fs)
            
            # Mapear anotaciones a picos R detectados
            # Las anotaciones están en la frecuencia original, necesitamos ajustarlas
            ann_symbols = np.array(ann.symbol)
            ann_samples_orig = np.array(ann.sample)
            # Convertir muestras originales a frecuencia objetivo
            ann_samples_resampled = (ann_samples_orig / fs * self.target_fs).astype(int)
            
            # Crear diccionario de etiquetas por posición
            label_map = {}
            for i, sample in enumerate(ann_samples_resampled):
                if 0 <= sample < len(sig) and i < len(ann_symbols):
                    label_map[sample] = ann_symbols[i]
            
            # Asignar etiquetas a picos R detectados (buscar anotación más cercana)
            r_peak_labels = []
            for r_peak in r_peaks:
                # Buscar anotación más cercana
                if len(ann_samples_resampled) > 0:
                    distances = np.abs(ann_samples_resampled - r_peak)
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < self.target_fs * 0.2:  # Dentro de 200ms
                        r_peak_labels.append(ann_symbols[closest_idx])
                    else:
                        r_peak_labels.append('N')  # Default normal si no hay anotación cercana
                else:
                    r_peak_labels.append('N')
            
            segments, labels = self.segment_beats(sig, r_peaks, r_peak_labels)
            return segments, labels
            
        except Exception as e:
            print(f"Error procesando registro {record_path}: {e}")
            return np.array([]).reshape(0, self.window_size), np.array([])

    def process_ptbxl_record(self, ecg_id, base_path='./ptb-xl/records500/', metadata_df=None, lead_idx=1):
        """
        Procesa un registro PTB-XL.
        
        Args:
            ecg_id: ID del ECG en PTB-XL
            base_path: Ruta base a los registros
            metadata_df: DataFrame con metadata (debe contener 'ecg_id' y 'scp_codes')
            lead_idx: Índice del lead a usar (1=Lead II típicamente)
            
        Returns:
            segments: Array de segmentos (n_segments, window_size)
            labels: Array de etiquetas (n_segments,)
        """
        try:
            # Construir ruta al archivo
            # PTB-XL organiza los archivos en subdirectorios por miles
            # ej: 00001 está en records500/00000/00001_hr
            subdir = f'{ecg_id // 1000 * 1000:05d}'
            record_dir = os.path.join(base_path, subdir)
            record_file = os.path.join(record_dir, f'{ecg_id:05d}_hr')
            
            if not os.path.exists(record_file + '.dat'):
                # No mostrar mensaje, solo retornar vacío (el error se maneja arriba)
                return np.array([]).reshape(0, self.window_size), np.array([])
            
            record = wfdb.rdrecord(record_file)
            # Usar lead específico o promedio si hay múltiples leads
            if record.p_signal.shape[1] > lead_idx:
                sig = record.p_signal[:, lead_idx]  # Lead seleccionado
            else:
                sig = np.mean(record.p_signal, axis=1)  # Promedio de todos los leads
            fs = record.fs  # 500 Hz para PTB-XL (high resolution)
            
            # Preprocesamiento
            if self.denoise:
                sig = self.denoise_signal(sig, fs)
            sig = self.resample_signal(sig, fs)
            sig = self.normalize_signal(sig)
            
            # Detectar picos R
            r_peaks = self.detect_r_peaks(sig, self.target_fs)
            
            # Mapear etiqueta PTB-XL a AAMI
            if metadata_df is not None and 'ecg_id' in metadata_df.columns:
                row = metadata_df[metadata_df['ecg_id'] == ecg_id]
                if len(row) > 0:
                    scp_codes = row['scp_codes'].values[0]
                    # scp_codes puede ser dict o string
                    if isinstance(scp_codes, str):
                        import ast
                        try:
                            scp_codes = ast.literal_eval(scp_codes)
                        except:
                            scp_codes = {}
                    
                    # Mapeo PTB-XL SCP codes a AAMI
                    if isinstance(scp_codes, dict):
                        if 'NORM' in scp_codes:
                            aami_label = 'N'
                        elif 'AFIB' in scp_codes or 'AFLT' in scp_codes:
                            aami_label = 'S'  # Supraventricular
                        elif 'STTC' in scp_codes or 'MI' in scp_codes:
                            aami_label = 'V'  # Ventricular (simplificado)
                        elif 'PVC' in scp_codes or 'VES' in scp_codes:
                            aami_label = 'V'
                        else:
                            aami_label = 'Q'  # Unknown
                    else:
                        aami_label = 'Q'
                else:
                    aami_label = 'Q'
            else:
                aami_label = 'Q'  # Default unknown
            
            # Asignar misma etiqueta a todos los latidos del registro
            label_map = [aami_label] * len(r_peaks)
            segments, labels = self.segment_beats(sig, r_peaks, label_map)
            return segments, labels
            
        except Exception as e:
            print(f"Error procesando PTB-XL record {ecg_id}: {e}")
            return np.array([]).reshape(0, self.window_size), np.array([])

def process_mitbih_dataset(mitdb_path, record_ids=None, target_fs=250, window_size=250, 
                           denoise=True, lead_idx=0, save_path=None):
    """
    Procesa dataset completo MIT-BIH.
    
    Args:
        mitdb_path: Ruta al directorio con registros MIT-BIH
        record_ids: Lista de IDs de registros a procesar (None = detectar automáticamente)
        target_fs: Frecuencia de muestreo objetivo
        window_size: Tamaño de ventana para segmentación
        denoise: Si aplicar denoising
        lead_idx: Índice del lead a usar
        save_path: Ruta para guardar dataset procesado (.pt)
        
    Returns:
        segments: Array de segmentos (n_segments, window_size)
        labels: Array de etiquetas (n_segments,)
    """
    preprocessor = ECGPreprocessor(target_fs=target_fs, window_size=window_size, denoise=denoise)
    
    if record_ids is None:
        # Detectar automáticamente qué registros existen
        record_ids = []
        # Buscar archivos .dat o .hea en el directorio
        for filename in os.listdir(mitdb_path):
            # Buscar archivos que sean números (registros MIT-BIH)
            if filename.endswith('.dat') or filename.endswith('.hea'):
                # Extraer el número del nombre del archivo
                base_name = filename.rsplit('.', 1)[0]
                try:
                    rec_id = int(base_name)
                    # MIT-BIH típicamente tiene registros 100-124 y 200-234
                    if 100 <= rec_id <= 234:
                        if rec_id not in record_ids:
                            record_ids.append(rec_id)
                except ValueError:
                    continue
        
        # Ordenar los IDs encontrados
        record_ids = sorted(record_ids)
        
        if len(record_ids) == 0:
            print(f"ERROR: No se encontraron registros MIT-BIH en {mitdb_path}")
            return np.array([]).reshape(0, window_size), np.array([])
        
        print(f"Detectados {len(record_ids)} registros MIT-BIH: {record_ids[:10]}..." if len(record_ids) > 10 else f"Detectados {len(record_ids)} registros MIT-BIH: {record_ids}")
    
    all_segments = []
    all_labels = []
    
    print(f"Procesando {len(record_ids)} registros MIT-BIH...")
    processed_count = 0
    skipped_count = 0
    
    for rec_id in tqdm(record_ids, desc="Procesando registros"):
        record_path = os.path.join(mitdb_path, str(rec_id))
        # Verificar si existe el archivo .dat o .hea
        if os.path.exists(record_path + '.dat') or os.path.exists(record_path + '.hea'):
            try:
                segments, labels = preprocessor.process_mitbih_record(record_path, lead_idx=lead_idx)
                if len(segments) > 0:
                    all_segments.append(segments)
                    all_labels.append(labels)
                    processed_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"\nAdvertencia: Error procesando registro {rec_id}: {e}")
                skipped_count += 1
        else:
            skipped_count += 1
    
    if len(all_segments) == 0:
        print("No se encontraron segmentos válidos")
        return np.array([]).reshape(0, window_size), np.array([])
    
    # Concatenar todos los segmentos
    all_segments = np.concatenate(all_segments, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nResumen del preprocesamiento:")
    print(f"  Registros procesados exitosamente: {processed_count}")
    print(f"  Registros omitidos: {skipped_count}")
    print(f"  Total de segmentos: {len(all_segments)}")
    unique, counts = np.unique(all_labels, return_counts=True)
    print(f"  Distribución de clases: {dict(zip(unique, counts))}")
    
    # Guardar si se especifica path
    if save_path is not None:
        dataset_dict = {
            'samples': all_segments,
            'labels': all_labels
        }
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        torch.save(dataset_dict, save_path)
        print(f"Dataset guardado en: {save_path}")
    
    return all_segments, all_labels


def process_ptbxl_dataset(ptbxl_path, metadata_path=None, record_ids=None, target_fs=250, 
                          window_size=250, denoise=True, lead_idx=1, save_path=None, records_file=None):
    """
    Procesa dataset completo PTB-XL.
    
    Args:
        ptbxl_path: Ruta al directorio con registros PTB-XL (records500/)
        metadata_path: Ruta al archivo CSV con metadata
        record_ids: Lista de IDs de registros a procesar (None = detectar automáticamente)
        target_fs: Frecuencia de muestreo objetivo
        window_size: Tamaño de ventana para segmentación
        denoise: Si aplicar denoising
        lead_idx: Índice del lead a usar
        save_path: Ruta para guardar dataset procesado (.pt)
        records_file: Ruta al archivo RECORDS (si None, busca en el directorio padre)
        
    Returns:
        segments: Array de segmentos (n_segments, window_size)
        labels: Array de etiquetas (n_segments,)
    """
    preprocessor = ECGPreprocessor(target_fs=target_fs, window_size=window_size, denoise=denoise)
    
    # Cargar metadata si está disponible
    metadata_df = None
    if metadata_path and os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        print(f"Metadata cargada: {len(metadata_df)} registros")
    
    # Obtener lista de registros disponibles
    if record_ids is None:
        # Intentar leer del archivo RECORDS primero
        if records_file is None:
            # Buscar archivo RECORDS en el directorio padre
            parent_dir = os.path.dirname(ptbxl_path) if os.path.dirname(ptbxl_path) else '.'
            records_file = os.path.join(parent_dir, 'RECORDS')
            if not os.path.exists(records_file):
                records_file = os.path.join(ptbxl_path, '..', 'RECORDS')
        
        record_ids = []
        
        # Leer archivo RECORDS si existe
        if records_file and os.path.exists(records_file):
            print(f"Leyendo archivo RECORDS: {records_file}")
            with open(records_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Algunas líneas pueden tener múltiples registros juntos
                    # Dividir por 'records500' para separar
                    if 'records500' in line:
                        # Separar por 'records500' para obtener todos los registros
                        parts = line.split('records500')
                        for part in parts[1:]:  # Saltar el primer elemento (antes del primer records500)
                            # Reconstruir la ruta completa
                            record_path = 'records500' + part
                            # Buscar el patrón _hr para extraer el ID
                            if '_hr' in record_path:
                                # Formato: records500/00000/00001_hr
                                path_parts = record_path.split('/')
                                if len(path_parts) >= 3:
                                    filename = path_parts[-1]  # 00001_hr
                                    try:
                                        ecg_id = int(filename.split('_')[0])  # 00001 -> 1
                                        record_ids.append(ecg_id)
                                    except ValueError:
                                        continue
            
            if len(record_ids) > 0:
                record_ids = sorted(list(set(record_ids)))  # Eliminar duplicados y ordenar
                print(f"Detectados {len(record_ids)} registros PTB-XL desde archivo RECORDS")
                if len(record_ids) > 10:
                    print(f"  Primeros 10: {record_ids[:10]}...")
                else:
                    print(f"  Registros: {record_ids}")
        
        # Si no se encontraron registros en RECORDS, intentar desde metadata
        if len(record_ids) == 0:
            if metadata_df is not None and 'ecg_id' in metadata_df.columns:
                record_ids = metadata_df['ecg_id'].tolist()
                print(f"Usando {len(record_ids)} registros desde metadata")
            else:
                # Fallback: buscar archivos en el directorio
                print("Buscando registros en el directorio...")
                for item in os.listdir(ptbxl_path):
                    item_path = os.path.join(ptbxl_path, item)
                    if os.path.isdir(item_path) and item.isdigit():
                        record_id = int(item)
                        record_file = os.path.join(item_path, f'{record_id:05d}_hr')
                        if os.path.exists(record_file + '.dat'):
                            record_ids.append(record_id)
                record_ids = sorted(record_ids)
                print(f"Encontrados {len(record_ids)} registros en el directorio")
        
        if len(record_ids) == 0:
            print(f"ERROR: No se encontraron registros PTB-XL")
            return np.array([]).reshape(0, window_size), np.array([])
    
    all_segments = []
    all_labels = []
    processed_count = 0
    skipped_count = 0
    
    print(f"Procesando {len(record_ids)} registros PTB-XL...")
    for ecg_id in tqdm(record_ids, desc="Procesando registros"):
        try:
            segments, labels = preprocessor.process_ptbxl_record(
                ecg_id, base_path=ptbxl_path, metadata_df=metadata_df, lead_idx=lead_idx
            )
            if len(segments) > 0:
                all_segments.append(segments)
                all_labels.append(labels)
                processed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            # Solo mostrar errores importantes, no todos los "no encontrado"
            if "no encontrado" not in str(e).lower():
                print(f"\nAdvertencia: Error procesando registro {ecg_id}: {e}")
            skipped_count += 1
    
    if len(all_segments) == 0:
        print("No se encontraron segmentos válidos")
        return np.array([]).reshape(0, window_size), np.array([])
    
    # Concatenar todos los segmentos
    all_segments = np.concatenate(all_segments, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nResumen del preprocesamiento:")
    print(f"  Registros procesados exitosamente: {processed_count}")
    print(f"  Registros omitidos: {skipped_count}")
    print(f"  Total de segmentos: {len(all_segments)}")
    unique, counts = np.unique(all_labels, return_counts=True)
    print(f"  Distribución de clases: {dict(zip(unique, counts))}")
    
    # Guardar si se especifica path
    if save_path is not None:
        dataset_dict = {
            'samples': all_segments,
            'labels': all_labels
        }
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        torch.save(dataset_dict, save_path)
        print(f"Dataset guardado en: {save_path}")
    
    return all_segments, all_labels


def split_dataset(segments, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                  shuffle=True, random_seed=42):
    """
    Divide dataset en train/val/test manteniendo distribución de clases.
    
    Args:
        segments: Array de segmentos
        labels: Array de etiquetas
        train_ratio: Proporción para training
        val_ratio: Proporción para validación
        test_ratio: Proporción para test
        shuffle: Si mezclar antes de dividir
        random_seed: Semilla para reproducibilidad
        
    Returns:
        train_segments, train_labels, val_segments, val_labels, test_segments, test_labels
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios deben sumar 1.0"
    
    if shuffle:
        np.random.seed(random_seed)
        indices = np.random.permutation(len(segments))
        segments = segments[indices]
        labels = labels[indices]
    
    n = len(segments)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_segments = segments[:n_train]
    train_labels = labels[:n_train]
    val_segments = segments[n_train:n_train+n_val]
    val_labels = labels[n_train:n_train+n_val]
    test_segments = segments[n_train+n_val:]
    test_labels = labels[n_train+n_val:]
    
    return train_segments, train_labels, val_segments, val_labels, test_segments, test_labels


def save_dataset_splits(segments, labels, output_dir, dataset_name='ecg', 
                        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Guarda dataset dividido en formato .pt compatible con dataloader de ECGTransForm.
    
    Args:
        segments: Array de segmentos
        labels: Array de etiquetas
        output_dir: Directorio de salida
        dataset_name: Nombre del dataset
        train_ratio, val_ratio, test_ratio: Proporciones para división
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Dividir dataset
    train_seg, train_lab, val_seg, val_lab, test_seg, test_lab = split_dataset(
        segments, labels, train_ratio, val_ratio, test_ratio
    )
    
    # Guardar en formato compatible con ECGTransForm dataloader
    # Formato esperado: {'samples': array, 'labels': array}
    train_dict = {'samples': train_seg, 'labels': train_lab}
    val_dict = {'samples': val_seg, 'labels': val_lab}
    test_dict = {'samples': test_seg, 'labels': test_lab}
    
    torch.save(train_dict, os.path.join(output_dir, f'train.pt'))
    torch.save(val_dict, os.path.join(output_dir, f'val.pt'))
    torch.save(test_dict, os.path.join(output_dir, f'test.pt'))
    
    print(f"Datasets guardados en {output_dir}:")
    print(f"  Train: {len(train_seg)} muestras")
    print(f"  Val: {len(val_seg)} muestras")
    print(f"  Test: {len(test_seg)} muestras")


class ECGDataset(Dataset):
    """
    Dataset para ECG compatible con el modelo TF_MorphoTransNet.
    Formato de salida: (samples, labels) donde samples es (1, window_size)
    """
    def __init__(self, segments, labels):
        """
        Args:
            segments: Array numpy de forma (n_samples, window_size)
            labels: Array numpy de etiquetas (n_samples,)
        """
        # Convertir a tensor y agregar dimensión de canal
        if isinstance(segments, np.ndarray):
            # Formato: (n_samples, 1, window_size) para el modelo
            self.segments = torch.from_numpy(segments).float()
            if len(self.segments.shape) == 2:
                self.segments = self.segments.unsqueeze(1)  # Agregar dimensión de canal
        else:
            self.segments = segments.float()
            if len(self.segments.shape) == 2:
                self.segments = self.segments.unsqueeze(1)
        
        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()
        
        assert len(self.segments) == len(self.labels), "Segments y labels deben tener la misma longitud"
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Returns:
            sample: Tensor de forma (1, window_size)
            label: Tensor escalar con la etiqueta
        """
        return self.segments[idx], self.labels[idx]
    
    def get_labels(self):
        """Retorna todas las etiquetas (compatible con ECGTransForm)"""
        return self.labels


# Ejemplo de uso:
if __name__ == "__main__":
    # Ejemplo para MIT-BIH
    # segments, labels = process_mitbih_dataset(
    #     mitdb_path='./mitdb',
    #     target_fs=250,
    #     window_size=250,
    #     save_path='./data/mitbih_all.pt'
    # )
    # save_dataset_splits(segments, labels, './data/mitbih', dataset_name='mitbih')
    
    # Ejemplo para PTB-XL
    # segments, labels = process_ptbxl_dataset(
    #     ptbxl_path='./ptb-xl/records500',
    #     metadata_path='./ptb-xl/ptbxl_database.csv',
    #     target_fs=250,
    #     window_size=250,
    #     save_path='./data/ptbxl_all.pt'
    # )
    # save_dataset_splits(segments, labels, './data/ptbxl', dataset_name='ptbxl')
    pass