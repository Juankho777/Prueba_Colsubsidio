"""
Main Script - Colsubsidio Churn Model

Script principal que ejecuta todo el pipeline del modelo de fuga.
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
import warnings

# Importar módulos del proyecto
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_utils import ModelTrainer
from src.business_logic import BusinessLogic

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_execution.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def main():
    """Ejecuta el pipeline completo del modelo."""
    
    logger.info("=== INICIANDO MODELO DE FUGA COLSUBSIDIO ===")
    
    try:
        # =====================================================================
        # PASO 1: CARGA DE DATOS
        # =====================================================================
        logger.info("PASO 1: Cargando datasets...")
        
        data_loader = DataLoader()
        datasets = data_loader.load_all_datasets()
        
        # Integrar datasets
        train_integrated, test_integrated = data_loader.integrate_datasets(datasets)
        
        # Análisis de distribución del target
        target_distribution = data_loader.get_target_distribution(train_integrated)
        logger.info(f"Desbalance de clases detectado - Ratio: {target_distribution.get('imbalance_ratio', 'N/A'):.1f}:1")
        
        # =====================================================================
        # PASO 2: FEATURE ENGINEERING
        # =====================================================================
        logger.info("PASO 2: Aplicando feature engineering...")
        
        feature_engineer = FeatureEngineer()
        
        # Crear variables derivadas
        train_enhanced = feature_engineer.apply_all_transformations(train_integrated)
        test_enhanced = feature_engineer.apply_all_transformations(test_integrated)
        
        # Validar features creadas
        validation_report = feature_engineer.validate_features(train_enhanced)
        logger.info(f"Features validadas - Total: {validation_report['total_features']}")
        
        # =====================================================================
        # PASO 3: PREPROCESAMIENTO
        # =====================================================================
        logger.info("PASO 3: Preprocesando datos para modelado...")
        
        preprocessor = DataPreprocessor()
        
        # Seleccionar features relevantes
        selected_features = preprocessor.select_features(train_enhanced)
        
        # Filtrar datasets con features seleccionadas + id + target
        train_features = train_enhanced[selected_features + ['id', 'Target']]
        test_features = test_enhanced[selected_features + ['id']]
        
        # Pipeline de preprocesamiento
        processed_data = preprocessor.prepare_model_data(train_features, test_features)
        
        X_train = processed_data['X_train']
        X_test = processed_data['X_test']
        y_train = processed_data['y_train']
        feature_names = processed_data['feature_names']
        test_ids = processed_data['test_ids']
        
        logger.info(f"Datos preparados - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # =====================================================================
        # PASO 4: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
        # =====================================================================
        logger.info("PASO 4: Entrenando modelos con diferentes estrategias...")
        
        model_trainer = ModelTrainer()
        
        # Crear split de validación para evaluación
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Entrenar múltiples estrategias
        results = model_trainer.train_multiple_strategies(
            X_train_split, y_train_split, X_val_split, y_val_split
        )
        
        # Comparar estrategias
        comparison_df = model_trainer.compare_strategies(results)
        logger.info("Comparación de estrategias:")
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        # Seleccionar mejor modelo
        best_model_result = model_trainer.select_best_model(results)
        best_model = best_model_result['model']
        
        # Entrenar modelo final con todos los datos de train
        logger.info("Entrenando modelo final con datos completos...")
        if best_model_result['strategy'] == 'Class Weights':
            class_weights = model_trainer.get_class_weights(y_train)
            final_model = model_trainer.get_model('random_forest', class_weights)
        elif best_model_result['strategy'] == 'Undersampling':
            X_train_final, y_train_final = model_trainer.create_undersampled_data(X_train, y_train)
            final_model = model_trainer.get_model('random_forest')
            final_model.fit(X_train_final, y_train_final)
        else:  # Oversampling
            X_train_final, y_train_final = model_trainer.create_oversampled_data(X_train, y_train)
            final_model = model_trainer.get_model('random_forest')
            final_model.fit(X_train_final, y_train_final)
        
        if best_model_result['strategy'] == 'Class Weights':
            final_model.fit(X_train, y_train)
        
        # =====================================================================
        # PASO 5: PREDICCIONES FINALES
        # =====================================================================
        logger.info("PASO 5: Generando predicciones finales...")
        
        # Predicciones de probabilidad
        final_predictions = final_model.predict_proba(X_test)[:, 1]
        
        logger.info(f"Predicciones generadas para {len(final_predictions)} clientes")
        logger.info(f"Rango de probabilidades: {final_predictions.min():.3f} - {final_predictions.max():.3f}")
        
        # =====================================================================
        # PASO 6: LÓGICA DE NEGOCIO Y SEGMENTACIÓN
        # =====================================================================
        logger.info("PASO 6: Aplicando lógica de negocio...")
        
        business_logic = BusinessLogic()
        
        # Crear segmentación de riesgo
        risk_segments, thresholds = business_logic.create_risk_segments(final_predictions)
        
        # Crear DataFrame final con resultados
        results_df = business_logic.create_client_scores_dataframe(
            test_ids, final_predictions, risk_segments, test_enhanced
        )
        
        # Generar recomendaciones de campaña
        campaign_recommendations = business_logic.generate_campaign_recommendations(
            risk_segments, test_enhanced
        )
        
        # Calcular impacto de negocio
        business_impact = business_logic.calculate_business_impact(campaign_recommendations)
        
        # =====================================================================
        # PASO 7: ANÁLISIS DE FEATURE IMPORTANCE
        # =====================================================================
        logger.info("PASO 7: Analizando importancia de variables...")
        
        feature_importance_df = model_trainer.get_feature_importance(final_model, feature_names)
        
        # Agrupar por categorías
        feature_groups = feature_engineer.get_feature_importance_groups()
        
        logger.info("Top 5 variables más importantes:")
        for _, row in feature_importance_df.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance_pct']:.1f}%")
        
        # =====================================================================
        # PASO 8: GENERAR RESUMEN EJECUTIVO
        # =====================================================================
        logger.info("PASO 8: Generando resumen ejecutivo...")
        
        executive_summary = business_logic.generate_executive_summary(
            campaign_recommendations, business_impact, best_model_result
        )
        
        # =====================================================================
        # PASO 9: GUARDAR RESULTADOS
        # =====================================================================
        logger.info("PASO 9: Guardando resultados...")
        
        # Crear directorio de outputs si no existe
        output_dir = Path("data/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar predicciones finales
        results_df.to_csv(output_dir / "final_predictions.csv", index=False)
        logger.info(f"Predicciones guardadas: {len(results_df)} registros")
        
        # Guardar feature importance
        feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
        
        # Guardar comparación de modelos
        comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
        
        # Guardar modelo entrenado
        model_trainer.save_model(final_model, output_dir / "best_model.pkl")
        
        # =====================================================================
        # PASO 10: MOSTRAR RESUMEN FINAL
        # =====================================================================
        logger.info("=== RESUMEN FINAL ===")
        logger.info(f"Mejor estrategia: {best_model_result['strategy']}")
        logger.info(f"AUC-ROC: {best_model_result['auc_roc']:.3f}")
        logger.info(f"Precision: {best_model_result['precision']:.3f}")
        logger.info(f"Recall: {best_model_result['recall']:.3f}")
        
        # Resumen de segmentación
        segment_counts = pd.Series(risk_segments).value_counts()
        logger.info("Distribución de riesgo:")
        for segment, count in segment_counts.items():
            logger.info(f"  {segment}: {count:,} clientes")
        
        # Resumen de negocio
        # Resumen de negocio
        logger.info(f"Framework status: {business_impact['framework_status']}")
        logger.info(f"Clientes prioritarios identificados: {business_impact['priority_clients_identified']:,}")
        logger.info(f"Próximo paso: {business_impact['next_steps'][0] if business_impact.get('next_steps') else 'Implementar estrategias'}")
        
        logger.info("=== MODELO COMPLETADO EXITOSAMENTE ===")
        
        return {
            'model': final_model,
            'predictions': results_df,
            'feature_importance': feature_importance_df,
            'business_impact': business_impact,
            'executive_summary': executive_summary
        }
        
    except Exception as e:
        logger.error(f"Error en ejecución del modelo: {e}")
        raise

if __name__ == "__main__":
    results = main()