"""
Business Logic Module - Colsubsidio Churn Model

Lógica de negocio para segmentación de riesgo y recomendaciones de campañas.
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
        self.business_config = self._load_config("business_rules.yaml")
        self.model_config = self._load_config("model_params.yaml")
        
        self.customer_value = self.business_config['customer_value']
        self.campaign_costs = self.business_config['campaign_costs']
        self.retention_strategies = self.business_config['retention_strategies']
        self.risk_thresholds = self.model_config['risk_segmentation']
        
    def _load_config(self, filename: str) -> dict:
        """Carga configuración desde archivo YAML."""
        with open(self.config_path / filename, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def create_risk_segments(self, scores: np.ndarray) -> tuple:
        """Segmenta clientes por nivel de riesgo basado en percentiles."""
        
        # Umbrales desde configuración
        p_high = self.risk_thresholds['thresholds']['high_risk']
        p_medium_high = self.risk_thresholds['thresholds']['medium_high'] 
        p_medium = self.risk_thresholds['thresholds']['medium']
        
        # Calcular umbrales reales
        threshold_high = np.percentile(scores, p_high)
        threshold_medium_high = np.percentile(scores, p_medium_high)
        threshold_medium = np.percentile(scores, p_medium)
        
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
    
    def generate_campaign_recommendations(self, risk_segments: list, 
                                        customer_data: pd.DataFrame = None) -> dict:
        """Genera recomendaciones específicas de campaña por segmento."""
        
        # Contar clientes por segmento
        segment_counts = pd.Series(risk_segments).value_counts()
        
        recommendations = {}
        
        for segment_name, strategy in self.retention_strategies.items():
            if segment_name.replace('_riesgo', '') in ['alto', 'medio_alto', 'medio', 'bajo']:
                
                # Mapear nombres de segmento
                segment_mapping = {
                    'alto_riesgo': 'Alto_Riesgo',
                    'medio_alto_riesgo': 'Medio_Alto_Riesgo', 
                    'medio_riesgo': 'Medio_Riesgo',
                    'bajo_riesgo': 'Bajo_Riesgo'
                }
                
                segment_label = segment_mapping.get(segment_name, segment_name)
                client_count = segment_counts.get(segment_label, 0)
                
                if client_count > 0:
                    # Calcular costos de campaña
                    # DESPUÉS (línea corregida):
                    cost_per_client = self.campaign_costs['by_risk_level'][segment_label]
                    total_budget = client_count * cost_per_client
                    
                    # Calcular ROI esperado
                    success_rate = strategy['success_rate']
                    avg_customer_value = self.customer_value['avg_annual_value']
                    expected_retention = int(client_count * success_rate)
                    expected_revenue = expected_retention * avg_customer_value
                    roi = (expected_revenue - total_budget) / total_budget if total_budget > 0 else 0
                    
                    recommendations[segment_label] = {
                        'client_count': client_count,
                        'action': strategy['action'],
                        'channels': strategy['channels'],
                        'offers': strategy['offers'],
                        'timeline': strategy['timeline'],
                        'cost_per_client': cost_per_client,
                        'total_budget': total_budget,
                        'success_rate': success_rate,
                        'expected_retention': expected_retention,
                        'expected_revenue': expected_revenue,
                        'roi': roi
                    }
        
        logger.info(f"Recomendaciones generadas para {len(recommendations)} segmentos")
        
        return recommendations
    
    def calculate_business_impact(self, recommendations: dict) -> dict:
        """Calcula el impacto total del negocio de las campañas."""
        
        total_clients = sum(rec['client_count'] for rec in recommendations.values())
        total_budget = sum(rec['total_budget'] for rec in recommendations.values())
        total_expected_revenue = sum(rec['expected_revenue'] for rec in recommendations.values())
        total_expected_retention = sum(rec['expected_retention'] for rec in recommendations.values())
        
        overall_roi = (total_expected_revenue - total_budget) / total_budget if total_budget > 0 else 0
        
        # Calcular reducción de churn esperada
        current_churn_rate = self.business_config['success_metrics']['monitoring_kpis']['monthly_churn_rate']
        retention_rate = total_expected_retention / total_clients if total_clients > 0 else 0
        churn_reduction = retention_rate * current_churn_rate
        
        impact = {
            'total_clients_targeted': total_clients,
            'total_investment': total_budget,
            'total_expected_revenue': total_expected_revenue,
            'total_clients_retained': total_expected_retention,
            'overall_roi': overall_roi,
            'current_churn_rate': current_churn_rate,
            'expected_churn_reduction': churn_reduction,
            'net_benefit': total_expected_revenue - total_budget
        }
        
        logger.info(f"Impacto calculado - ROI: {overall_roi:.2f}x, "
                   f"Inversión: ${total_budget:,.0f}")
        
        return impact
    
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
        """Genera resumen ejecutivo para presentación."""
        
        # Métricas del modelo
        model_summary = {
            'algorithm': 'Random Forest',
            'auc_roc': model_performance.get('auc_roc', 0),
            'precision': model_performance.get('precision', 0),
            'recall': model_performance.get('recall', 0),
            'strategy': model_performance.get('strategy', 'Unknown')
        }
        
        # Segmentación de clientes
        segmentation_summary = {}
        for segment, rec in recommendations.items():
            segmentation_summary[segment] = {
                'clients': rec['client_count'],
                'budget': rec['total_budget'],
                'action': rec['action']
            }
        
        # KPIs clave
        key_metrics = {
            'total_clients_at_risk': sum(rec['client_count'] for rec in recommendations.values()),
            'total_investment_required': business_impact['total_investment'],
            'expected_roi': business_impact['overall_roi'],
            'expected_churn_reduction': business_impact['expected_churn_reduction'],
            'net_business_value': business_impact['net_benefit']
        }
        
        # Próximos pasos
        next_steps = [
            "Implementar campaña urgente para clientes de alto riesgo",
            "Configurar scoring automático mensual", 
            "Desarrollar dashboard ejecutivo de monitoreo",
            "Evaluar efectividad de campañas después de 30 días"
        ]
        
        executive_summary = {
            'model_performance': model_summary,
            'client_segmentation': segmentation_summary,
            'key_business_metrics': key_metrics,
            'recommended_next_steps': next_steps,
            'success_criteria': {
                'min_roi_achieved': business_impact['overall_roi'] >= 3.0,
                'churn_reduction_target': business_impact['expected_churn_reduction'] >= 0.15,
                'model_performance_adequate': model_summary['auc_roc'] >= 0.75
            }
        }
        
        logger.info("Resumen ejecutivo generado")
        
        return executive_summary
    
    def validate_business_rules(self, customer_data: pd.DataFrame) -> dict:
        """Valida que los datos cumplan con reglas de negocio."""
        
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
        
        # Validar coherencia de cupo vs saldo
        if all(col in customer_data.columns for col in ['Saldo', 'Limite.Cupo']):
            invalid_cupo = customer_data[customer_data['Saldo'] > customer_data['Limite.Cupo']]
            if len(invalid_cupo) > 0:
                validation_results['warnings'].append(
                    f"Saldo mayor que límite de cupo: {len(invalid_cupo)} registros"
                )
        
        logger.info(f"Validación completada - Errores: {len(validation_results['validation_errors'])}, "
                   f"Advertencias: {len(validation_results['warnings'])}")
        
        return validation_results