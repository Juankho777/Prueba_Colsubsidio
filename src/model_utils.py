"""
Model Utilities Module - Colsubsidio Churn Model

Funciones para entrenamiento, evaluación y manejo de desbalance de clases.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score
import joblib
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Maneja entrenamiento y evaluación de modelos."""
    
    def __init__(self, config_path: str = "config/"):
        self.config_path = Path(config_path)
        self.config = self._load_config("model_params.yaml")
        self.model_config = self.config['models']
        self.balance_config = self.config['class_balance']
        self.validation_config = self.config['validation']
        
    def _load_config(self, filename: str) -> dict:
        """Carga configuración desde archivo YAML."""
        with open(self.config_path / filename, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_class_weights(self, y: pd.Series) -> dict:
        """Calcula pesos balanceados para las clases."""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, weights))
        
        logger.info(f"Pesos calculados - Clase 0: {class_weight_dict[0]:.3f}, Clase 1: {class_weight_dict[1]:.3f}")
        return class_weight_dict
    
    def create_undersampled_data(self, X: np.ndarray, y: pd.Series) -> tuple:
        """Crea dataset balanceado usando undersampling."""
        
        # Resetear índices para evitar problemas
        y_reset = y.reset_index(drop=True)
        
        # Obtener posiciones por clase
        pos_class_0 = np.where(y_reset == 0)[0]
        pos_class_1 = np.where(y_reset == 1)[0]
        
        # Configurar ratio de undersampling
        ratio_multiplier = self.balance_config['undersampling']['ratio_multiplier']
        n_minority = len(pos_class_1)
        n_majority_sample = min(n_minority * ratio_multiplier, len(pos_class_0))
        
        # Muestreo aleatorio de clase mayoritaria
        np.random.seed(self.balance_config['undersampling']['random_state'])
        pos_class_0_sample = np.random.choice(pos_class_0, size=n_majority_sample, replace=False)
        
        # Combinar posiciones
        balanced_positions = np.concatenate([pos_class_0_sample, pos_class_1])
        np.random.shuffle(balanced_positions)
        
        # Crear datasets balanceados
        X_balanced = X[balanced_positions]
        y_balanced = y_reset.iloc[balanced_positions]
        
        logger.info(f"Undersampling - Nuevo tamaño: {len(X_balanced)} (ratio {n_majority_sample/n_minority:.1f}:1)")
        
        return X_balanced, y_balanced
    
    def create_oversampled_data(self, X: np.ndarray, y: pd.Series) -> tuple:
        """Crea dataset balanceado usando oversampling simple."""
        
        y_reset = y.reset_index(drop=True)
        
        # Obtener posiciones por clase
        pos_class_0 = np.where(y_reset == 0)[0]
        pos_class_1 = np.where(y_reset == 1)[0]
        
        # Configurar multiplicador de oversampling
        multiplier = self.balance_config['oversampling']['minority_multiplier']
        
        # Replicar clase minoritaria
        pos_minority_replicated = np.tile(pos_class_1, multiplier)
        
        # Combinar con clase mayoritaria
        all_positions = np.concatenate([pos_class_0, pos_minority_replicated])
        np.random.seed(self.balance_config['oversampling']['random_state'])
        np.random.shuffle(all_positions)
        
        # Crear dataset oversampled
        X_oversampled = X[all_positions]
        y_oversampled = y_reset.iloc[all_positions]
        
        logger.info(f"Oversampling - Nuevo tamaño: {len(X_oversampled)}")
        
        return X_oversampled, y_oversampled
    
    def get_model(self, model_name: str, class_weights: dict = None) -> object:
        """Crea instancia del modelo configurado."""
        
        if model_name == 'random_forest':
            params = self.model_config['random_forest'].copy()
            if class_weights:
                params['class_weight'] = class_weights
            return RandomForestClassifier(**params)
        
        elif model_name == 'logistic_regression':
            params = self.model_config['logistic_regression'].copy()
            if class_weights:
                params['class_weight'] = class_weights
            return LogisticRegression(**params)
        
        else:
            raise ValueError(f"Modelo no soportado: {model_name}")
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: pd.Series, 
                      strategy_name: str) -> dict:
        """Evalúa modelo con métricas completas."""
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas básicas
        metrics = {
            'strategy': strategy_name,
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Precision at K (top 10% por defecto)
        k_pct = self.config['risk_segmentation']['precision_at_k']
        top_k = int(len(y_pred_proba) * k_pct)
        top_indices = np.argsort(y_pred_proba)[-top_k:]
        metrics['precision_at_k'] = y_test.iloc[top_indices].mean()
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics['confusion_matrix'] = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        
        # Predicciones para uso posterior
        metrics['predictions'] = y_pred_proba
        metrics['model'] = model
        
        logger.info(f"{strategy_name} - AUC: {metrics['auc_roc']:.3f}, "
                   f"Precision: {metrics['precision']:.3f}, "
                   f"Recall: {metrics['recall']:.3f}")
        
        return metrics
    
    def train_multiple_strategies(self, X_train: np.ndarray, y_train: pd.Series,
                                X_test: np.ndarray, y_test: pd.Series) -> list:
        """Entrena modelos con diferentes estrategias de balanceo."""
        
        results = []
        
        # 1. Estrategia: Class Weights
        logger.info("Entrenando con class weights...")
        class_weights = self.get_class_weights(y_train)
        model_weighted = self.get_model('random_forest', class_weights)
        model_weighted.fit(X_train, y_train)
        
        result_weighted = self.evaluate_model(
            model_weighted, X_test, y_test, 'Class Weights'
        )
        results.append(result_weighted)
        
        # 2. Estrategia: Undersampling
        logger.info("Entrenando con undersampling...")
        X_under, y_under = self.create_undersampled_data(X_train, y_train)
        model_under = self.get_model('random_forest')
        model_under.fit(X_under, y_under)
        
        result_under = self.evaluate_model(
            model_under, X_test, y_test, 'Undersampling'
        )
        results.append(result_under)
        
        # 3. Estrategia: Oversampling
        logger.info("Entrenando con oversampling...")
        X_over, y_over = self.create_oversampled_data(X_train, y_train)
        model_over = self.get_model('random_forest')
        model_over.fit(X_over, y_over)
        
        result_over = self.evaluate_model(
            model_over, X_test, y_test, 'Oversampling'
        )
        results.append(result_over)
        
        return results
    
    def select_best_model(self, results: list) -> dict:
        """Selecciona el mejor modelo basado en AUC-ROC."""
        
        best_result = max(results, key=lambda x: x['auc_roc'])
        
        logger.info(f"Mejor modelo: {best_result['strategy']} "
                   f"(AUC: {best_result['auc_roc']:.3f})")
        
        return best_result
    
    def cross_validate_model(self, model, X: np.ndarray, y: pd.Series) -> dict:
        """Realiza validación cruzada del modelo."""
        
        cv_scores = cross_val_score(
            model, X, y,
            cv=self.validation_config['cv_folds'],
            scoring=self.validation_config['cv_scoring']
        )
        
        cv_results = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        logger.info(f"CV Score: {cv_results['mean_score']:.3f} (+/- {cv_results['std_score']*2:.3f})")
        
        return cv_results
    
    def save_model(self, model, filepath: str):
        """Guarda modelo entrenado."""
        joblib.dump(model, filepath)
        logger.info(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath: str):
        """Carga modelo previamente entrenado."""
        model = joblib.load(filepath)
        logger.info(f"Modelo cargado desde: {filepath}")
        return model
    
    def get_feature_importance(self, model, feature_names: list) -> pd.DataFrame:
        """Extrae feature importance del modelo."""
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_,
                'importance_pct': model.feature_importances_ * 100
            }).sort_values('importance', ascending=False)
            
            logger.info(f"Feature importance extraída para {len(feature_names)} variables")
            return importance_df
        
        else:
            logger.warning("Modelo no soporta feature importance")
            return pd.DataFrame()
    
    def compare_strategies(self, results: list) -> pd.DataFrame:
        """Compara resultados de diferentes estrategias."""
        
        comparison_data = []
        for result in results:
            comparison_data.append({
                'Strategy': result['strategy'],
                'AUC_ROC': result['auc_roc'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1_Score': result['f1_score'],
                'Precision_at_K': result['precision_at_k']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        logger.info("Comparación de estrategias completada")
        return comparison_df