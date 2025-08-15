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
    """
    Carga y valida datasets del proyecto con detección automática de contexto.
    
    Esta clase detecta automáticamente si está siendo ejecutada desde:
    - main.py (desde raíz del proyecto)
    - notebooks individuales (desde notebooks/)
    - entornos de testing o Docker
    
    Y ajusta las rutas automáticamente para cada contexto.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.execution_context = self._detect_execution_context()
        self.config_path = self._resolve_config_path(config_path)
        self.config = self._load_config_with_fallback("model_params.yaml")
        self.data_paths = self._get_context_appropriate_paths()
        self.file_names = self.config['data']['files']
        
        logger.info(f"DataLoader inicializado - Contexto: {self.execution_context}")
        logger.debug(f"Rutas de datos: {self.data_paths}")
        
    def _detect_execution_context(self) -> str:
        """
        Detecta automáticamente el contexto de ejecución.
        
        Returns:
            str: 'notebook', 'main_script', 'test', o 'unknown'
        """
        try:
            # Detectar si estamos en Jupyter/IPython
            shell = get_ipython()
            if shell is not None:
                if 'ZMQInteractiveShell' in str(type(shell)):
                    return 'notebook'
                elif 'TerminalInteractiveShell' in str(type(shell)):
                    return 'ipython'
        except NameError:
            pass
        
        # Analizar directorio actual y stack de llamadas
        current_dir = Path.cwd()
        
        # Si estamos en notebooks/ o subdirectorio
        if current_dir.name == 'notebooks' or 'notebooks' in current_dir.parts:
            return 'notebook'
        
        # Si estamos en tests/ o subdirectorio
        if current_dir.name == 'tests' or 'tests' in current_dir.parts:
            return 'test'
        
        # Verificar si main.py está en el directorio actual
        if (current_dir / 'main.py').exists():
            return 'main_script'
        
        # Verificar stack de llamadas para detectar pytest
        import inspect
        for frame_info in inspect.stack():
            if 'pytest' in frame_info.filename or 'test_' in frame_info.filename:
                return 'test'
        
        return 'unknown'
    
    def _resolve_config_path(self, config_path: Optional[str]) -> Path:
        """
        Resuelve la ruta del directorio de configuración.
        
        Args:
            config_path: Ruta opcional proporcionada por el usuario
            
        Returns:
            Path: Ruta resuelta al directorio de configuración
        """
        if config_path:
            return Path(config_path)
        
        # Rutas candidatas según el contexto
        candidate_paths = []
        
        if self.execution_context == 'notebook':
            candidate_paths = [
                Path('../config'),
                Path('../../config'),  # Por si hay subdirectorios
                Path('config'),
            ]
        elif self.execution_context == 'test':
            candidate_paths = [
                Path('../config'),
                Path('config'),
                Path('./config'),
            ]
        else:  # main_script o unknown
            candidate_paths = [
                Path('config'),
                Path('./config'),
                Path('../config'),
            ]
        
        # Buscar el primer directorio que exista y contenga model_params.yaml
        for path in candidate_paths:
            if path.exists() and (path / 'model_params.yaml').exists():
                logger.debug(f"Config encontrado en: {path.absolute()}")
                return path
        
        # Si no encontramos nada, usar el primero de la lista
        logger.warning(f"No se encontró config válido. Usando: {candidate_paths[0]}")
        return candidate_paths[0]
    
    def _load_config_with_fallback(self, filename: str) -> dict:
        """
        Carga configuración con múltiples intentos de fallback.
        
        Args:
            filename: Nombre del archivo de configuración
            
        Returns:
            dict: Configuración cargada
            
        Raises:
            FileNotFoundError: Si no se puede encontrar el archivo en ninguna ubicación
        """
        config_file = self.config_path / filename
        
        # Lista de rutas a intentar
        fallback_paths = [
            config_file,
            Path('config') / filename,
            Path('../config') / filename,
            Path('../../config') / filename,
            Path('./') / filename,
        ]
        
        last_error = None
        
        for attempt_path in fallback_paths:
            try:
                if attempt_path.exists():
                    with open(attempt_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        logger.debug(f"Configuración cargada desde: {attempt_path.absolute()}")
                        return config
            except Exception as e:
                last_error = e
                logger.debug(f"Error intentando cargar {attempt_path}: {e}")
                continue
        
        # Si llegamos aquí, no pudimos cargar el archivo
        error_msg = f"No se pudo cargar {filename} desde ninguna ubicación. Rutas intentadas: {[str(p) for p in fallback_paths]}"
        if last_error:
            error_msg += f"\nÚltimo error: {last_error}"
        
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    def _get_context_appropriate_paths(self) -> dict:
        """
        Obtiene las rutas de datos apropiadas según el contexto.
        
        Returns:
            dict: Diccionario con rutas de datos
        """
        # Intentar usar configuración dinámica si está disponible
        if 'path_detection' in self.config and 'contexts' in self.config['path_detection']:
            contexts = self.config['path_detection']['contexts']
            
            if self.execution_context in contexts:
                return contexts[self.execution_context]
            elif self.execution_context == 'notebook' and 'notebook' in contexts:
                return contexts['notebook']
            elif self.execution_context in ['main_script', 'unknown'] and 'main_script' in contexts:
                return contexts['main_script']
        
        # Fallback a configuración estática
        base_paths = self.config['data']['paths']
        
        if self.execution_context == 'notebook':
            # Ajustar rutas para notebooks
            return {
                'raw_data': '../data/raw/',
                'processed_data': '../data/processed/',
                'outputs': '../data/outputs/'
            }
        else:
            # Usar rutas desde configuración para main_script
            return {
                'raw_data': base_paths.get('raw_data', 'data/raw/'),
                'processed_data': base_paths.get('processed_data', 'data/processed/'),
                'outputs': base_paths.get('outputs', 'data/outputs/')
            }
    
    def _clean_financial_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Convierte columnas de texto con formato monetario a números.
        
        Args:
            df: DataFrame a limpiar
            columns: Lista de nombres de columnas financieras
            
        Returns:
            pd.DataFrame: DataFrame con columnas financieras limpiadas
        """
        df_clean = df.copy()
        replacements = self.config['preprocessing']['replacements']
        
        for col in columns:
            if col in df_clean.columns:
                original_type = df_clean[col].dtype
                
                # Convertir a string y limpiar formato de texto
                df_clean[col] = df_clean[col].astype(str)
                for char in replacements:
                    df_clean[col] = df_clean[col].str.replace(char, '', regex=False)
                
                # Manejar valores especiales
                df_clean[col] = df_clean[col].replace(['nan', 'NaN', 'None', ''], np.nan)
                
                # Convertir a numérico
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                logger.debug(f"Columna {col} convertida de {original_type} a numeric")
        
        return df_clean
    
    def _verify_file_exists(self, file_path: Path, file_description: str) -> None:
        """
        Verifica que un archivo exista y lanza error descriptivo si no.
        
        Args:
            file_path: Ruta del archivo a verificar
            file_description: Descripción del archivo para el mensaje de error
            
        Raises:
            FileNotFoundError: Si el archivo no existe
        """
        if not file_path.exists():
            # Intentar encontrar archivos similares para sugerir
            parent_dir = file_path.parent
            if parent_dir.exists():
                similar_files = list(parent_dir.glob(f"*{file_path.suffix}"))
                suggestion = f"\nArchivos encontrados en {parent_dir}: {[f.name for f in similar_files]}" if similar_files else f"\nDirectorio {parent_dir} está vacío"
            else:
                suggestion = f"\nDirectorio {parent_dir} no existe"
            
            error_msg = f"No se encuentra {file_description}: {file_path}{suggestion}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    
    def load_train_data(self) -> pd.DataFrame:
        """
        Carga el dataset de entrenamiento con validaciones.
        
        Returns:
            pd.DataFrame: Dataset de entrenamiento limpio
        """
        file_path = Path(self.data_paths['raw_data']) / self.file_names['train']
        self._verify_file_exists(file_path, "archivo de entrenamiento")
        
        try:
            df = pd.read_csv(
                file_path,
                sep=self.config['data']['separator'],
                encoding=self.config['data']['encoding']
            )
            
            logger.info(f"Train cargado: {len(df):,} registros, {len(df.columns)} columnas")
            
            # Limpiar variables financieras
            financial_cols = self.config['preprocessing']['financial_columns']
            df_clean = self._clean_financial_columns(df, financial_cols)
            
            # Validaciones básicas
            if len(df_clean) == 0:
                raise ValueError("El dataset de entrenamiento está vacío")
            
            if 'id' not in df_clean.columns:
                logger.warning("No se encontró columna 'id' en el dataset de entrenamiento")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cargando dataset de entrenamiento desde {file_path}: {e}")
            raise
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Carga el dataset de prueba con validaciones.
        
        Returns:
            pd.DataFrame: Dataset de prueba limpio
        """
        file_path = Path(self.data_paths['raw_data']) / self.file_names['test']
        self._verify_file_exists(file_path, "archivo de prueba")
        
        try:
            df = pd.read_csv(
                file_path,
                sep=self.config['data']['separator'],
                encoding=self.config['data']['encoding']
            )
            
            logger.info(f"Test cargado: {len(df):,} registros, {len(df.columns)} columnas")
            
            # Limpiar variables financieras
            financial_cols = self.config['preprocessing']['financial_columns']
            df_clean = self._clean_financial_columns(df, financial_cols)
            
            # Validaciones básicas
            if len(df_clean) == 0:
                raise ValueError("El dataset de prueba está vacío")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cargando dataset de prueba desde {file_path}: {e}")
            raise
    
    def load_demograficas_data(self) -> pd.DataFrame:
        """
        Carga datos demográficos desde Excel con validaciones.
        
        Returns:
            pd.DataFrame: Dataset demográfico
        """
        file_path = Path(self.data_paths['raw_data']) / self.file_names['demograficas']
        self._verify_file_exists(file_path, "archivo demográfico")
        
        try:
            df = pd.read_excel(file_path)
            logger.info(f"Demográficas cargado: {len(df):,} registros, {len(df.columns)} columnas")
            
            if len(df) == 0:
                raise ValueError("El dataset demográfico está vacío")
            
            return df
            
        except Exception as e:
            logger.error(f"Error cargando datos demográficos desde {file_path}: {e}")
            raise
    
    def load_subsidios_data(self) -> pd.DataFrame:
        """
        Carga datos de subsidios desde Excel con validaciones.
        
        Returns:
            pd.DataFrame: Dataset de subsidios
        """
        file_path = Path(self.data_paths['raw_data']) / self.file_names['subsidios']
        self._verify_file_exists(file_path, "archivo de subsidios")
        
        try:
            df = pd.read_excel(file_path)
            logger.info(f"Subsidios cargado: {len(df):,} registros, {len(df.columns)} columnas")
            
            if len(df) == 0:
                raise ValueError("El dataset de subsidios está vacío")
                
            return df
            
        except Exception as e:
            logger.error(f"Error cargando datos de subsidios desde {file_path}: {e}")
            raise
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Carga todos los datasets necesarios con manejo robusto de errores.
        
        Returns:
            Dict[str, pd.DataFrame]: Diccionario con todos los datasets
        """
        datasets = {}
        
        try:
            logger.info(f"Iniciando carga de datasets - Contexto: {self.execution_context}")
            
            # Cargar datasets uno por uno con manejo individual de errores
            datasets['train'] = self.load_train_data()
            datasets['test'] = self.load_test_data()
            datasets['demograficas'] = self.load_demograficas_data()
            datasets['subsidios'] = self.load_subsidios_data()
            
            logger.info("Todos los datasets cargados exitosamente")
            
            # Validaciones cruzadas
            self._validate_dataset_consistency(datasets)
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error en carga de datasets: {e}")
            logger.error(f"Contexto de ejecución: {self.execution_context}")
            logger.error(f"Rutas utilizadas: {self.data_paths}")
            raise
    
    def _validate_dataset_consistency(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """
        Valida la consistencia entre datasets.
        
        Args:
            datasets: Diccionario con los datasets cargados
        """
        # Verificar que todos tienen la columna 'id'
        for name, df in datasets.items():
            if 'id' not in df.columns:
                logger.warning(f"Dataset {name} no tiene columna 'id'")
        
        # Verificar solapamiento entre train y test
        if 'train' in datasets and 'test' in datasets:
            train_ids = set(datasets['train']['id'].unique())
            test_ids = set(datasets['test']['id'].unique())
            overlap = train_ids.intersection(test_ids)
            
            if overlap:
                logger.warning(f"Solapamiento entre train y test: {len(overlap)} IDs")
            else:
                logger.info("No hay solapamiento entre train y test")
    
    def integrate_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Integra train y test con datos complementarios.
        
        Args:
            datasets: Diccionario con datasets individuales
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train y test integrados
        """
        logger.info("Iniciando integración de datasets...")
        
        try:
            # Merge train con datos adicionales
            train_integrated = (
                datasets['train']
                .merge(datasets['demograficas'], on='id', how='left', suffixes=('', '_demo'))
                .merge(datasets['subsidios'], on='id', how='left', suffixes=('', '_subs'))
            )
            
            # Merge test con datos adicionales
            test_integrated = (
                datasets['test']
                .merge(datasets['demograficas'], on='id', how='left', suffixes=('', '_demo'))
                .merge(datasets['subsidios'], on='id', how='left', suffixes=('', '_subs'))
            )
            
            # Validar que no perdimos registros
            if len(train_integrated) != len(datasets['train']):
                logger.warning(f"Train perdió registros en integración: {len(datasets['train'])} -> {len(train_integrated)}")
            
            if len(test_integrated) != len(datasets['test']):
                logger.warning(f"Test perdió registros en integración: {len(datasets['test'])} -> {len(test_integrated)}")
            
            logger.info(f"Integración completada - Train: {len(train_integrated):,}, Test: {len(test_integrated):,}")
            
            return train_integrated, test_integrated
            
        except Exception as e:
            logger.error(f"Error en integración de datasets: {e}")
            raise
    
    def get_target_distribution(self, train_data: pd.DataFrame) -> Dict:
        """
        Analiza la distribución de la variable target.
        
        Args:
            train_data: Dataset de entrenamiento
            
        Returns:
            Dict: Estadísticas de distribución del target
        """
        if 'Target' not in train_data.columns:
            logger.warning("Variable Target no encontrada")
            return {}
        
        target_counts = train_data['Target'].value_counts()
        target_props = train_data['Target'].value_counts(normalize=True)
        
        distribution = {
            'counts': target_counts.to_dict(),
            'proportions': target_props.to_dict(),
            'imbalance_ratio': target_counts[0] / target_counts[1] if 1 in target_counts else None,
            'total_samples': len(train_data),
            'missing_target': train_data['Target'].isnull().sum()
        }
        
        logger.info(f"Distribución target - No Fuga: {target_props[0]:.1%}, Fuga: {target_props[1]:.1%}")
        if distribution['imbalance_ratio'] and distribution['imbalance_ratio'] > 10:
            logger.warning(f"Desbalance extremo detectado: {distribution['imbalance_ratio']:.1f}:1")
        
        return distribution
    
    def get_diagnostics(self) -> Dict:
        """
        Obtiene información diagnóstica del DataLoader.
        
        Returns:
            Dict: Información diagnóstica
        """
        return {
            'execution_context': self.execution_context,
            'config_path': str(self.config_path.absolute()),
            'data_paths': self.data_paths,
            'current_directory': str(Path.cwd()),
            'config_exists': (self.config_path / 'model_params.yaml').exists(),
            'data_directories_exist': {
                name: Path(path).exists() for name, path in self.data_paths.items()
            }
        }