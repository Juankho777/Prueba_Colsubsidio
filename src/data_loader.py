"""
Data Loader Module - Colsubsidio Churn Model

Maneja la carga de datasets desde archivos CSV y Excel.
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
    """Carga y valida datasets del proyecto."""
    
    def __init__(self, config_path: str = "config/"):
        self.config_path = Path(config_path)
        self.data_config = self._load_config("model_params.yaml")
        self.schema_config = self._load_config("data_schema.yaml")
        
        self.data_paths = self.data_config['data']['paths']
        self.file_names = self.data_config['data']['files']
        
    def _load_config(self, filename: str) -> dict:
        """Lee archivos YAML de configuración."""
        config_file = self.config_path / filename
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _clean_financial_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Convierte columnas de texto con formato monetario a números."""
        df_clean = df.copy()
        replacements = self.data_config['preprocessing']['replacements']
        
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
        
        df = pd.read_csv(
            file_path,
            sep=self.data_config['data']['separator'],
            encoding=self.data_config['data']['encoding']
        )
        
        logger.info(f"Train cargado: {len(df):,} registros")
        
        # Limpiar variables financieras
        financial_cols = self.data_config['preprocessing']['financial_columns']
        df_clean = self._clean_financial_columns(df, financial_cols)
        
        return df_clean
    
    def load_test_data(self) -> pd.DataFrame:
        """Carga el dataset de prueba."""
        file_path = Path(self.data_paths['raw_data']) / self.file_names['test']
        
        df = pd.read_csv(
            file_path,
            sep=self.data_config['data']['separator'],
            encoding=self.data_config['data']['encoding']
        )
        
        logger.info(f"Test cargado: {len(df):,} registros")
        
        # Limpiar variables financieras
        financial_cols = self.data_config['preprocessing']['financial_columns']
        df_clean = self._clean_financial_columns(df, financial_cols)
        
        return df_clean
    
    def load_demograficas_data(self) -> pd.DataFrame:
        """Carga datos demográficos desde Excel."""
        file_path = Path(self.data_paths['raw_data']) / self.file_names['demograficas']
        
        df = pd.read_excel(file_path)
        logger.info(f"Demográficas cargado: {len(df):,} registros")
        
        return df
    
    def load_subsidios_data(self) -> pd.DataFrame:
        """Carga datos de subsidios desde Excel."""
        file_path = Path(self.data_paths['raw_data']) / self.file_names['subsidios']
        
        df = pd.read_excel(file_path)
        logger.info(f"Subsidios cargado: {len(df):,} registros")
        
        return df
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Carga todos los datasets de una vez."""
        datasets = {}
        
        try:
            datasets['train'] = self.load_train_data()
            datasets['test'] = self.load_test_data()
            datasets['demograficas'] = self.load_demograficas_data()
            datasets['subsidios'] = self.load_subsidios_data()
            
            logger.info("Todos los datasets cargados exitosamente")
            return datasets
            
        except Exception as e:
            logger.error(f"Error cargando datasets: {e}")
            raise
    
    def integrate_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Integra train y test con datos complementarios."""
        
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