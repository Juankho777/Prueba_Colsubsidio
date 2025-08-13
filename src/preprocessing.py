"""
Preprocessing Module - Colsubsidio Churn Model

Funciones para limpieza y preparación de datos antes del modelado.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Maneja la limpieza y preparación de datos."""
    
    def __init__(self, config_path: str = "config/"):
        self.config_path = Path(config_path)
        self.config = self._load_config("model_params.yaml")
        self.preprocessing_config = self.config['preprocessing']
        self.validation_config = self.config['validation']
        
    def _load_config(self, filename: str) -> dict:
        """Carga configuración desde archivo YAML."""
        with open(self.config_path / filename, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maneja valores faltantes según estrategia configurada."""
        strategy = self.preprocessing_config['missing_strategy']
        df_clean = df.copy()
        
        if strategy == 'fill_zero':
            # Rellenar NaN con 0 para variables numéricas
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
            
        elif strategy == 'drop':
            df_clean = df_clean.dropna()
            
        elif strategy == 'median':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
                df_clean[numeric_cols].median()
            )
        
        logger.info(f"Valores faltantes manejados con estrategia: {strategy}")
        return df_clean
    
    def encode_categorical_variables(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> tuple:
        """Codifica variables categóricas usando Label Encoding."""
        train_encoded = train_df.copy()
        test_encoded = test_df.copy() if test_df is not None else None
        
        # Identificar columnas categóricas
        categorical_cols = train_encoded.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['id']]
        
        encoders = {}
        
        for col in categorical_cols:
            if col in train_encoded.columns:
                encoder = LabelEncoder()
                
                # Ajustar encoder con datos de train
                train_encoded[col] = train_encoded[col].astype(str)
                train_encoded[col] = encoder.fit_transform(train_encoded[col])
                encoders[col] = encoder
                
                # Aplicar a test si existe
                if test_encoded is not None and col in test_encoded.columns:
                    test_encoded[col] = test_encoded[col].astype(str)
                    # Manejar valores no vistos en train
                    try:
                        test_encoded[col] = encoder.transform(test_encoded[col])
                    except ValueError:
                        # Si hay categorías nuevas, asignar valor por defecto
                        known_values = set(encoder.classes_)
                        test_encoded[col] = test_encoded[col].apply(
                            lambda x: encoder.transform([x])[0] if x in known_values else -1
                        )
        
        logger.info(f"Variables categóricas codificadas: {len(categorical_cols)}")
        
        if test_encoded is not None:
            return train_encoded, test_encoded, encoders
        return train_encoded, encoders
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> tuple:
        """Escala características usando StandardScaler."""
        scaler = StandardScaler()
        
        # Ajustar y transformar train
        X_train_scaled = scaler.fit_transform(X_train)
        
        results = [X_train_scaled, scaler]
        
        # Transformar test si existe
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            results.insert(-1, X_test_scaled)  # Insertar antes del scaler
        
        logger.info("Características escaladas con StandardScaler")
        return tuple(results)
    
    def prepare_train_features(self, train_df: pd.DataFrame) -> tuple:
        """Prepara features y target del dataset de entrenamiento."""
        
        # Separar features y target
        X = train_df.drop(['Target'], axis=1)
        y = train_df['Target']
        
        logger.info(f"Features preparadas - Train: {len(X)} registros, {len(X.columns)} variables")
        
        return X, y
    
    def prepare_model_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """Pipeline completo de preparación de datos para modelado."""
        
        # 1. Manejar valores faltantes
        train_clean = self.handle_missing_values(train_df)
        test_clean = self.handle_missing_values(test_df)
        
        # 2. Codificar variables categóricas
        train_encoded, test_encoded, encoders = self.encode_categorical_variables(
            train_clean, test_clean
        )
        
        # 3. Separar features y target
        X_train, y_train = self.prepare_train_features(train_encoded)
        
        # 4. Preparar features para test
        X_test = test_encoded.drop(['id'], axis=1, errors='ignore')
        
        # 5. Escalar características
        X_train_scaled, X_test_scaled, scaler = self.scale_features(
            X_train.drop(['id'], axis=1, errors='ignore'), 
            X_test=X_test
        )
        
        # Preparar resultado
        processed_data = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'feature_names': X_train.drop(['id'], axis=1, errors='ignore').columns.tolist(),
            'encoders': encoders,
            'scaler': scaler,
            'test_ids': test_encoded['id'].values if 'id' in test_encoded.columns else None
        }
        
        logger.info("Pipeline de preprocesamiento completado")
        return processed_data
    
    def get_data_quality_report(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """Genera reporte de calidad de datos."""
        
        report = {
            'dataset': dataset_name,
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicated_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Variables con más del 50% de missing
        high_missing = {
            col: pct for col, pct in report['missing_percentage'].items() 
            if pct > 50
        }
        
        if high_missing:
            logger.warning(f"Variables con >50% missing en {dataset_name}: {list(high_missing.keys())}")
        
        return report
    
    def select_features(self, train_df: pd.DataFrame) -> list:
        """Selecciona features relevantes para el modelo."""
        
        # Variables financieras clave
        financial_vars = [
            'Saldo', 'Limite.Cupo', 'Edad.Mora', 'Vr.Mora', 
            'Pagos.Mes.Ant', 'Vtas.Mes.Ant', 'Saldos.Mes.Ant'
        ]
        
        # Variables demográficas
        demographic_vars = [
            'edad', 'segmento', 'estrato', 'contrato', 'nivel_educativo'
        ]
        
        # Variables de subsidios
        subsidy_vars = [
            'cuota_monetaria', 'sub_vivenda', 'bono_lonchera'
        ]
        
        # Combinar todas las variables disponibles
        all_vars = financial_vars + demographic_vars + subsidy_vars
        available_vars = [var for var in all_vars if var in train_df.columns]
        
        logger.info(f"Features seleccionadas: {len(available_vars)}")
        return available_vars