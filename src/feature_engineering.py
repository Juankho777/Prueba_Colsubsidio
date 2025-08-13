"""
Feature Engineering Module - Colsubsidio Churn Model

Crea variables derivadas con lógica de negocio crediticio.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Genera variables derivadas para mejorar la predicción."""
    
    def __init__(self, config_path: str = "config/"):
        self.config_path = Path(config_path)
        self.config = self._load_config("model_params.yaml")
        self.feature_config = self.config['feature_engineering']
        
    def _load_config(self, filename: str) -> dict:
        """Carga configuración desde archivo YAML."""
        with open(self.config_path / filename, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def create_utilization_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula ratio de utilización del cupo de crédito."""
        df_new = df.copy()
        
        config = self.feature_config['derived_features']['utilization_ratio']
        numerator = config['numerator']
        denominator = config['denominator']
        default_val = config['default_value']
        
        # Evitar división por cero
        df_new['utilization_ratio'] = np.where(
            df_new[denominator] > 0,
            df_new[numerator] / df_new[denominator],
            default_val
        )
        
        return df_new
    
    def create_payment_behavior(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula índice de comportamiento de pago."""
        df_new = df.copy()
        
        config = self.feature_config['derived_features']['payment_behavior']
        numerator = config['numerator']
        denominator = config['denominator']
        adjustment = config['adjustment']
        default_val = config['default_value']
        
        # Ratio de pagos vs saldos con ajuste para evitar división por cero
        df_new['payment_behavior'] = np.where(
            (df_new[denominator] > 0) & (df_new[numerator] >= 0),
            df_new[numerator] / (df_new[denominator] + adjustment),
            default_val
        )
        
        return df_new
    
    def create_financial_stress(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea índice de estrés financiero basado en múltiples indicadores."""
        df_new = df.copy()
        
        config = self.feature_config['derived_features']['financial_stress']
        stress_score = 0
        
        for component in config['components']:
            condition = component['condition']
            
            if condition == "Edad.Mora > 0":
                stress_score += (df_new['Edad.Mora'] > 0).astype(int)
            elif condition == "Vr.Mora > 0":
                stress_score += (df_new['Vr.Mora'] > 0).astype(int)
            elif condition == "utilization_ratio > 0.8":
                if 'utilization_ratio' in df_new.columns:
                    stress_score += (df_new['utilization_ratio'] > 0.8).astype(int)
        
        df_new['financial_stress'] = stress_score
        
        return df_new
    
    def create_client_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mide nivel de actividad transaccional del cliente."""
        df_new = df.copy()
        
        config = self.feature_config['derived_features']['client_activity']
        activity_score = 0
        
        for component in config['components']:
            condition = component['condition']
            
            if condition == "Vtas.Mes.Ant > 0":
                activity_score += (df_new['Vtas.Mes.Ant'] > 0).astype(int)
            elif condition == "Pagos.Mes.Ant > 0":
                activity_score += (df_new['Pagos.Mes.Ant'] > 0).astype(int)
        
        df_new['client_activity'] = activity_score
        
        return df_new
    
    def create_benefits_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula índice de beneficios Colsubsidio."""
        df_new = df.copy()
        
        config = self.feature_config['derived_features']['benefits_index']
        components = config['components']
        
        # Sumar todos los componentes de beneficios
        benefits_sum = 0
        for component in components:
            if component in df_new.columns:
                benefits_sum += df_new[component].fillna(0)
        
        df_new['benefits_index'] = benefits_sum
        
        return df_new
    
    def create_inactive_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identifica clientes completamente inactivos."""
        df_new = df.copy()
        
        config = self.feature_config['derived_features']['is_inactive']
        conditions = config['conditions']
        
        # Cliente inactivo si cumple todas las condiciones
        is_inactive = True
        for condition in conditions:
            if condition == "Saldo == 0":
                is_inactive = is_inactive & (df_new['Saldo'] == 0)
            elif condition == "Vtas.Mes.Ant == 0":
                is_inactive = is_inactive & (df_new['Vtas.Mes.Ant'] == 0)
        
        df_new['is_inactive'] = is_inactive.astype(int)
        
        return df_new
    
    def create_risk_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categoriza clientes por perfil de riesgo crediticio."""
        df_new = df.copy()
        
        # Crear categorías basadas en días de mora
        df_new['risk_profile'] = np.select([
            df_new['Edad.Mora'] == 0,
            df_new['Edad.Mora'] <= 30,
            df_new['Edad.Mora'] <= 90,
            df_new['Edad.Mora'] > 90
        ], [
            'Sin_Mora', 
            'Mora_Temprana', 
            'Mora_Media', 
            'Mora_Severa'
        ], default='Sin_Mora')
        
        return df_new
    
    def create_utilization_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categoriza la utilización del cupo en rangos."""
        df_new = df.copy()
        
        if 'utilization_ratio' not in df_new.columns:
            logger.warning("utilization_ratio no encontrada, creándola primero")
            df_new = self.create_utilization_ratio(df_new)
        
        # Crear categorías de utilización
        df_new['util_category'] = pd.cut(
            df_new['utilization_ratio'],
            bins=[0, 0.1, 0.5, 0.8, float('inf')],
            labels=['Muy_Baja', 'Baja', 'Media', 'Alta'],
            include_lowest=True
        )
        
        return df_new
    
    def apply_all_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica todas las transformaciones de feature engineering."""
        logger.info("Iniciando feature engineering...")
        
        df_transformed = df.copy()
        
        # Rellenar NaN con 0 para cálculos
        numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
        df_transformed[numeric_cols] = df_transformed[numeric_cols].fillna(0)
        
        # Aplicar todas las transformaciones en secuencia
        transformations = [
            self.create_utilization_ratio,
            self.create_payment_behavior,
            self.create_financial_stress,
            self.create_client_activity,
            self.create_benefits_index,
            self.create_inactive_flag,
            self.create_risk_profile,
            self.create_utilization_category
        ]
        
        for transform in transformations:
            try:
                df_transformed = transform(df_transformed)
                logger.info(f"Aplicada transformación: {transform.__name__}")
            except Exception as e:
                logger.warning(f"Error en {transform.__name__}: {e}")
        
        # Contar nuevas variables creadas
        original_cols = set(df.columns)
        new_cols = set(df_transformed.columns) - original_cols
        
        logger.info(f"Feature engineering completado. Nuevas variables: {len(new_cols)}")
        logger.info(f"Variables creadas: {list(new_cols)}")
        
        return df_transformed
    
    def get_feature_importance_groups(self) -> dict:
        """Define grupos de variables para análisis de importancia."""
        
        groups = {
            'financial_original': [
                'Saldo', 'Limite.Cupo', 'Edad.Mora', 'Vr.Mora',
                'Pagos.Mes.Ant', 'Vtas.Mes.Ant', 'Saldos.Mes.Ant'
            ],
            'financial_derived': [
                'utilization_ratio', 'payment_behavior', 'financial_stress',
                'client_activity', 'is_inactive'
            ],
            'demographic': [
                'edad', 'segmento', 'estrato', 'contrato', 'nivel_educativo'
            ],
            'benefits': [
                'cuota_monetaria', 'sub_vivenda', 'bono_lonchera', 'benefits_index'
            ],
            'categorical_derived': [
                'risk_profile', 'util_category'
            ]
        }
        
        return groups
    
    def validate_features(self, df: pd.DataFrame) -> dict:
        """Valida que las features creadas tengan sentido de negocio."""
        
        validation_report = {
            'total_features': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'infinite_values': {},
            'negative_values': {},
            'feature_ranges': {}
        }
        
        # Verificar valores infinitos en features numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                validation_report['infinite_values'][col] = inf_count
        
        # Verificar valores negativos donde no deberían existir
        positive_only_features = [
            'utilization_ratio', 'benefits_index', 'client_activity', 
            'financial_stress', 'is_inactive'
        ]
        
        for feature in positive_only_features:
            if feature in df.columns:
                negative_count = (df[feature] < 0).sum()
                if negative_count > 0:
                    validation_report['negative_values'][feature] = negative_count
        
        # Rangos de features clave
        key_features = ['utilization_ratio', 'payment_behavior', 'financial_stress']
        for feature in key_features:
            if feature in df.columns:
                validation_report['feature_ranges'][feature] = {
                    'min': df[feature].min(),
                    'max': df[feature].max(),
                    'mean': df[feature].mean(),
                    'std': df[feature].std()
                }
        
        logger.info("Validación de features completada")
        return validation_report