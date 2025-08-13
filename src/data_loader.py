"""
Data Loader Module - Colsubsidio Churn Model

Maneja la carga de datasets desde archivos CSV y Excel con detección automática de contexto.
"""

import pandas as pd
import numpy as np
import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Carga y valida datasets del proyecto con detección automática de contexto."""
    
    def __init__(self, config_path: str = "config/"):
        self.config_path = Path(config_path)
        self.context = self._detect_execution_context()
        self.config = self._load_config("model_params.yaml")
        self.data_paths = self._get_appropriate_paths()
        self.file_names = self.config['data']['files']
        
        logger.info(f"DataLoader iniciado en contexto: {self.context}")
        logger.info(f"Usando rutas: {self.data_paths}")
        
    def _detect_execution_context(self) -> str:
        """Detecta si estamos ejecutando desde notebook o script principal."""
        try:
            # Verificar si estamos en un notebook
            if 'ipykernel' in str(type(get_ipython())):
                return "notebook"
        except NameError:
            pass
        
        # Verificar el directorio actual
        current_dir = os.getcwd()
        if current_dir.endswith('notebooks'):
            return "notebook"
        elif 'notebooks' in current_dir:
            return "notebook"
        else:
            return "main_script"
    
    def _load_config(self, filename: str) -> dict:
        """Lee archivos YAML de configuración con manejo robusto de rutas."""
        config_file = self.config_path / filename
        
        # Si no encuentra el archivo, intentar con rutas alternativas
        if not config_file.exists():
            alternative_paths = [
                Path("../config") / filename,  # Para notebooks
                Path("config") / filename,      # Para main
                Path(".") / filename,          # Directorio actual
            ]
            
            for alt_path in alternative_paths:
                if alt_path.exists():
                    config_file = alt_path
                    break
            else:
                raise FileNotFoundError(f"No se encuentra {filename} en ninguna de las rutas: {[str(p) for p in [self.config_path / filename] + alternative_paths]}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_appropriate_paths(self) -> dict:
        """Obtiene las rutas apropiadas según el contexto de ejecución."""
        if self.context == "notebook":
            # Usar rutas para notebooks
            return self.config['path_detection']['contexts']['notebook']
        else:
            # Usar rutas para script principal
            return self.config['path_detection']['contexts']['main_script']
    
    def _clean_financial_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Convierte columnas de texto con formato monetario a números."""
        df_clean = df.copy()
        replacements = self.config['preprocessing']['replacements']
        
        for col in columns:
            if col in df_clean.columns:
                # Limpiar formato de texto
                df_clean[col] = df_clean[col].astype(str)
                for char in replacements:
                    df_clean[col] = df_clean[col].str.replace(char, '')
                
                # Convertir a numérico
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        return df_clean
    
    def load_train_data(self) -> pd.DataFrame:
        """Carga el dataset de entrenamiento."""
        file_path = Path(self.data_paths['raw_data']) / self.file_names['train']
        
        if not file_path.exists():
            raise FileNotFoundError(f"No se encuentra el archivo train en: {file_path}")
        
        df = pd.read_csv(
            file_path,
            sep=self.config['data']['separator'],
            encoding=self.config['data']['encoding']
        )
        
        logger.info(f"Train cargado: {len(df):,} registros desde {file_path}")
        
        # Limpiar variables financieras
        financial_cols = self.config['preprocessing']['financial_columns']
        df_clean = self._clean_financial_columns(df, financial_cols)
        
        return df_clean
    
    def load_test_data(self) -> pd.DataFrame:
        """Carga el dataset de prueba."""
        file_path = Path(self.data_paths['raw_data']) / self.file_names['test']
        
        if not file_path.exists():
            raise FileNotFoundError(f"No se encuentra el archivo test en: {file_path}")
        
        df = pd.read_csv(
            file_path,
            sep=self.config['data']['separator'],
            encoding=self.config['data']['encoding']
        )
        
        logger.info(f"Test cargado: {len(df):,} registros desde {file_path}")
        
        # Limpiar variables financieras
        financial_cols = self.config['preprocessing']['financial_columns']
        df_clean = self._clean_financial_columns(df, financial_cols)
        
        return df_clean
    
    def load_demograficas_data(self) -> pd.DataFrame:
        """Carga datos demográficos desde Excel."""
        file_path = Path(self.data_paths['raw_data']) / self.file_names['demograficas']
        
        if not file_path.exists():
            raise FileNotFoundError(f"No se encuentra el archivo demográficas en: {file_path}")
        
        df = pd.read_excel(file_path)
        logger.info(f"Demográficas cargado: {len(df):,} registros desde {file_path}")
        
        return df
    
    def load_subsidios_data(self) -> pd.DataFrame:
        """Carga datos de subsidios desde Excel."""
        file_path = Path(self.data_paths['raw_data']) / self.file_names['subsidios']
        
        if not file_path.exists():
            raise FileNotFoundError(f"No se encuentra el archivo subsidios en: {file_path}")
        
        df = pd.read_excel(file_path)
        logger.info(f"Subsidios cargado: {len(df):,} registros desde {file_path}")
        
        return df
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Carga todos los datasets de una vez."""
        datasets = {}
        
        try:
            logger.info(f"Cargando todos los datasets en contexto: {self.context}")
            
            datasets['train'] = self.load_train_data()
            datasets['test'] = self.load_test_data()
            datasets['demograficas'] = self.load_demograficas_data()
            datasets['subsidios'] = self.load_subsidios_data()
            
            logger.info("Todos los datasets cargados exitosamente")
            return datasets
            
        except Exception as e:
            logger.error(f"Error cargando datasets: {e}")
            logger.error(f"Rutas utilizadas: {self.data_paths}")
            logger.error(f"Contexto detectado: {self.context}")
            raise
    
    def integrate_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Integra train y test con datos complementarios."""
        
        logger.info("Integrando datasets...")
        
        # Merge train con datos adicionales
        train_integrated = (
            datasets['train']
            .merge(datasets['demograficas'], on='id', how='left')
            .merge(datasets['subsidios'], on='id', how='left')
        )
        
        # Merge test con datos adicionales
        test_integrated = (
            datasets['test']
            .merge(datasets['demograficas'], on='id', how='left')
            .merge(datasets['subsidios'], on='id', how='left')
        )
        
        logger.info(f"Integración completada - Train: {len(train_integrated)}, Test: {len(test_integrated)}")
        
        return train_integrated, test_integrated
    
    def get_target_distribution(self, train_data: pd.DataFrame) -> Dict:
        """Analiza la distribución de la variable target."""
        if 'Target' not in train_data.columns:
            logger.warning("Variable Target no encontrada")
            return {}
        
        target_counts = train_data['Target'].value_counts()
        target_props = train_data['Target'].value_counts(normalize=True)
        
        distribution = {
            'counts': target_counts.to_dict(),
            'proportions': target_props.to_dict(),
            'imbalance_ratio': target_counts[0] / target_counts[1] if 1 in target_counts else None
        }
        
        logger.info(f"Distribución target - No Fuga: {target_props[0]:.1%}, Fuga: {target_props[1]:.1%}")
        
        return distribution
    
    def verify_data_paths(self) -> dict:
        """Verifica que todas las rutas de datos existan."""
        verification = {}
        
        for path_name, path_value in self.data_paths.items():
            path_obj = Path(path_value)
            verification[path_name] = {
                'path': str(path_obj),
                'exists': path_obj.exists(),
                'is_dir': path_obj.is_dir() if path_obj.exists() else False
            }
        
        # Verificar archivos específicos
        for file_name, file_value in self.file_names.items():
            file_path = Path(self.data_paths['raw_data']) / file_value
            verification[f"file_{file_name}"] = {
                'path': str(file_path),
                'exists': file_path.exists(),
                'size_mb': file_path.stat().st_size / (1024*1024) if file_path.exists() else 0
            }
        
        return verification