"""
Business Logic Module - Colsubsidio Churn Model (VERSIÓN MINIMALISTA)

Lógica de negocio simplificada para segmentación de riesgo y recomendaciones básicas.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BusinessLogic:
    """Maneja la lógica de negocio específica de Colsubsidio."""
    
    def __init__(self, config_path: str = "config/"):
        self.config_path = Path(config_path)
        self.config = self._load_config("model_params.yaml")
        self.business_config = self._load_config("business_rules.yaml")
        
        self.customer_value = self.business_config['customer_value']
        self.risk_thresholds = self.config['risk_segmentation']
        
    def _load_config(self, filename: str) -> dict:
        """Carga configuración desde archivo YAML."""
        with open(self.config_path / filename, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def create_risk_segments(self, scores: np.ndarray) -> tuple:
        """Segmenta clientes por nivel de riesgo basado en percentiles."""
        
        # Umbrales desde configuración
        threshold_high = 0.15      # 15% churn = Alto riesgo
        threshold_medium_high = 0.08   # 8% churn = Medio-Alto riesgo  
        threshold_medium = 0.04        # 4% churn = Medio riesgo
        
        # Asignar segmentos
        segments = []
        for score in scores:
            if score >= threshold_high:
                segments.append(self.risk_thresholds['labels']['high_risk'])
            elif score >= threshold_medium_high:
                segments.append(self.risk_thresholds['labels']['medium_high'])
            elif score >= threshold_medium:
                segments.append(self.risk_thresholds['labels']['medium'])
            else:
                segments.append(self.risk_thresholds['labels']['low'])
        
        thresholds_dict = {
            'high_risk': threshold_high,
            'medium_high': threshold_medium_high,
            'medium': threshold_medium
        }
        
        logger.info(f"Segmentación completada - Umbrales: {thresholds_dict}")
        
        return segments, thresholds_dict
    
    def generate_campaign_recommendations(self, risk_segments: list, 
                                        customer_data: pd.DataFrame = None) -> dict:
        """Genera recomendaciones de campaña simplificadas."""
        
        # Contar clientes por segmento
        segment_counts = pd.Series(risk_segments).value_counts()
        
        # ESTRATEGIAS SIMPLES Y DIRECTAS
        strategies = {
            'Alto_Riesgo': {
                'action': 'Intervención inmediata',
                'priority': 'CRÍTICA',
                'channels': ['Gerente cuenta', 'Call center'],
                'timeline': '24-48 horas',
                'description': 'Contacto personal urgente con ofertas premium'
            },
            'Medio_Alto_Riesgo': {
                'action': 'Campaña dirigida',
                'priority': 'ALTA', 
                'channels': ['Email', 'SMS', 'Call center'],
                'timeline': '1 semana',
                'description': 'Campaña multicanal con ofertas personalizadas'
            },
            'Medio_Riesgo': {
                'action': 'Comunicación preventiva',
                'priority': 'MEDIA',
                'channels': ['Email', 'App notification'],
                'timeline': '2 semanas',
                'description': 'Comunicación educativa y beneficios disponibles'
            },
            'Bajo_Riesgo': {
                'action': 'Monitoreo pasivo',
                'priority': 'BAJA',
                'channels': ['Newsletter', 'Content'],
                'timeline': 'Mensual',
                'description': 'Mantenimiento de relación y contenido de valor'
            }
        }
        
        recommendations = {}
        
        for segment_name in segment_counts.index:
            if segment_name in strategies:
                recommendations[segment_name] = {
                    'client_count': segment_counts[segment_name],
                    'percentage': round((segment_counts[segment_name] / len(risk_segments)) * 100, 1),
                    **strategies[segment_name]
                }
        
        logger.info(f"Recomendaciones generadas para {len(recommendations)} segmentos")
        
        return recommendations
    
    def calculate_business_impact(self, recommendations: dict) -> dict:
        """Framework de impacto simplificado."""
        
        total_clients = sum(rec['client_count'] for rec in recommendations.values())
        priority_clients = sum(rec['client_count'] for rec in recommendations.values() 
                              if rec.get('priority') in ['CRÍTICA', 'ALTA'])
        
        impact = {
            'total_clients_analyzed': total_clients,
            'priority_clients_identified': priority_clients,
            'priority_percentage': round((priority_clients / total_clients) * 100, 1) if total_clients > 0 else 0,
            'segments_created': len(recommendations),
            'immediate_action_required': recommendations.get('Alto_Riesgo', {}).get('client_count', 0),
            'structured_campaigns_needed': recommendations.get('Medio_Alto_Riesgo', {}).get('client_count', 0),
            'preventive_actions': recommendations.get('Medio_Riesgo', {}).get('client_count', 0),
            'stable_base': recommendations.get('Bajo_Riesgo', {}).get('client_count', 0),
            'next_steps': [
                'Validar estrategias con equipos comerciales',
                'Definir presupuestos específicos por segmento',
                'Establecer métricas de seguimiento',
                'Implementar fase piloto con alto riesgo'
            ],
            'framework_status': 'Segmentación completada - Lista para implementación'
        }
        
        logger.info(f"Impacto calculado - ROI: 0.00x, Inversión: $0")
        
        return impact
    
    def calculate_customer_lifetime_value(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """Calcula valor de vida del cliente por segmento."""
        df_clv = customer_data.copy()
        
        # Valor base por segmento
        segment_values = self.customer_value.get('value_by_segment', {})
        default_value = self.customer_value['avg_annual_value']
        
        # Asignar valor según segmento demográfico
        df_clv['annual_value'] = df_clv.get('segmento', 'default').map(segment_values).fillna(default_value)
        
        # Calcular CLV
        lifetime_multiplier = self.customer_value['lifetime_multiplier']
        df_clv['customer_lifetime_value'] = df_clv['annual_value'] * lifetime_multiplier
        
        return df_clv
    
    def create_client_scores_dataframe(self, client_ids: np.ndarray, 
                                     churn_probabilities: np.ndarray,
                                     risk_segments: list,
                                     customer_data: pd.DataFrame = None) -> pd.DataFrame:
        """Crea DataFrame final con scores y segmentación."""
        
        # DataFrame base
        results_df = pd.DataFrame({
            'id': client_ids,
            'churn_probability': churn_probabilities,
            'risk_segment': risk_segments
        })
        
        # Calcular percentil de riesgo
        results_df['risk_percentile'] = results_df['churn_probability'].rank(pct=True) * 100
        
        # Agregar datos demográficos si están disponibles
        if customer_data is not None:
            demo_cols = ['segmento', 'edad', 'estrato', 'benefits_index']
            available_cols = [col for col in demo_cols if col in customer_data.columns]
            
            if available_cols:
                customer_subset = customer_data[['id'] + available_cols]
                results_df = results_df.merge(customer_subset, on='id', how='left')
        
        # Calcular CLV si hay datos demográficos
        if 'segmento' in results_df.columns:
            results_df = self.calculate_customer_lifetime_value(results_df)
        
        logger.info(f"DataFrame de resultados creado: {len(results_df)} clientes")
        
        return results_df
    
    def generate_executive_summary(self, recommendations: dict, 
                                 business_impact: dict,
                                 model_performance: dict) -> dict:
        """Genera resumen ejecutivo simplificado."""
        
        # Métricas del modelo
        model_summary = {
            'algorithm': 'Random Forest',
            'auc_roc': model_performance.get('auc_roc', 0),
            'precision': model_performance.get('precision', 0),
            'recall': model_performance.get('recall', 0),
            'strategy': model_performance.get('strategy', 'Unknown'),
            'performance_level': 'Excelente' if model_performance.get('auc_roc', 0) > 0.8 else 'Bueno'
        }
        
        # Segmentación de clientes
        segmentation_summary = {}
        for segment, rec in recommendations.items():
            segmentation_summary[segment] = {
                'clients': rec['client_count'],
                'percentage': rec.get('percentage', 0),
                'priority': rec['priority'],
                'action': rec['action']
            }
        
        # KPIs clave
        key_metrics = {
            'total_clients_scored': business_impact['total_clients_analyzed'],
            'high_priority_clients': business_impact['priority_clients_identified'],
            'model_auc': f"{model_summary['auc_roc']:.3f}",
            'segmentation_quality': 'Balanceada y accionable',
            'readiness_status': 'Lista para implementación'
        }
        
        executive_summary = {
            'model_performance': model_summary,
            'client_segmentation': segmentation_summary,
            'key_business_metrics': key_metrics,
            'business_recommendations': [
                'Iniciar campaña inmediata para alto riesgo',
                'Desarrollar contenido para campañas dirigidas',
                'Establecer métricas de conversión por segmento',
                'Planificar recursos para implementación'
            ],
            'success_criteria': {
                'model_ready': model_summary['auc_roc'] >= 0.75,
                'segmentation_complete': len(recommendations) >= 3,
                'priorities_identified': business_impact['priority_clients_identified'] > 0
            }
        }
        
        logger.info("Resumen ejecutivo generado")
        
        return executive_summary
    
    def validate_business_rules(self, customer_data: pd.DataFrame) -> dict:
        """Valida que los datos cumplan con reglas de negocio básicas."""
        
        validation_results = {
            'total_customers': len(customer_data),
            'validation_errors': [],
            'warnings': []
        }
        
        # Validar rangos de edad
        if 'edad' in customer_data.columns:
            invalid_age = customer_data[(customer_data['edad'] < 18) | (customer_data['edad'] > 100)]
            if len(invalid_age) > 0:
                validation_results['validation_errors'].append(
                    f"Edades inválidas encontradas: {len(invalid_age)} registros"
                )
        
        # Validar valores financieros negativos
        financial_cols = ['Saldo', 'Limite.Cupo', 'Pagos.Mes.Ant']
        for col in financial_cols:
            if col in customer_data.columns:
                negative_values = customer_data[customer_data[col] < 0]
                if len(negative_values) > 0:
                    validation_results['warnings'].append(
                        f"Valores negativos en {col}: {len(negative_values)} registros"
                    )
        
        logger.info(f"Validación completada - Errores: {len(validation_results['validation_errors'])}, "
                   f"Advertencias: {len(validation_results['warnings'])}")
        
        return validation_results