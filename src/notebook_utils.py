"""
Notebook Utilities - Colsubsidio Churn Model

"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

# Configurar logging
logger = logging.getLogger(__name__)

def get_notebook_data_loader():
    """
    Wrapper para usar DataLoader desde notebooks sin problemas de rutas.
    
    Esta función maneja automáticamente el cambio de directorio necesario
    para que DataLoader funcione correctamente desde notebooks.
    
    Returns:
        DataLoader: Instancia configurada de DataLoader
        
    Example:
        >>> from src.notebook_utils import get_notebook_data_loader
        >>> data_loader = get_notebook_data_loader()
        >>> datasets = data_loader.load_all_datasets()
    """
    from src.data_loader import DataLoader
    
    # Guardar directorio actual
    original_cwd = os.getcwd()
    
    try:
        # Si estamos en notebooks/, cambiar al directorio raíz
        if original_cwd.endswith('notebooks') or 'notebooks' in Path(original_cwd).parts:
            project_root = Path(original_cwd).parent
            os.chdir(project_root)
            logger.info(f"Cambiado a directorio raíz: {project_root}")
        
        # Crear DataLoader (ahora funcionará con rutas correctas)
        loader = DataLoader()
        logger.info("DataLoader creado exitosamente desde notebook")
        
        return loader
        
    except Exception as e:
        logger.error(f"Error creando DataLoader desde notebook: {e}")
        raise
    finally:
        # Siempre restaurar directorio original
        os.chdir(original_cwd)

def load_data_for_notebook() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Función completa para cargar e integrar datos desde notebooks.
    
    Esta función hace todo el proceso de carga e integración en una sola llamada,
    perfecta para análisis exploratorio.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train y test integrados
        
    Example:
        >>> from src.notebook_utils import load_data_for_notebook
        >>> train_data, test_data = load_data_for_notebook()
    """
    try:
        # Usar DataLoader
        data_loader = get_notebook_data_loader()
        
        # Cargar todos los datasets
        datasets = data_loader.load_all_datasets()
        
        # Integrar datos
        train_integrated, test_integrated = data_loader.integrate_datasets(datasets)
        
        logger.info(f"Datos cargados - Train: {len(train_integrated):,}, Test: {len(test_integrated):,}")
        
        return train_integrated, test_integrated
        
    except Exception as e:
        logger.error(f"Error en carga completa de datos: {e}")
        raise

def get_target_info(train_data: pd.DataFrame) -> Dict:
    """
    Análisis rápido del target para notebooks.
    
    Args:
        train_data: Dataset de entrenamiento
        
    Returns:
        Dict: Información del target
    """
    if 'Target' not in train_data.columns:
        return {'error': 'Target no encontrado'}
    
    target_counts = train_data['Target'].value_counts()
    target_props = train_data['Target'].value_counts(normalize=True)
    
    info = {
        'total_samples': len(train_data),
        'no_fuga': target_counts[0],
        'fuga': target_counts[1] if 1 in target_counts else 0,
        'no_fuga_pct': target_props[0],
        'fuga_pct': target_props[1] if 1 in target_props else 0,
        'imbalance_ratio': target_counts[0] / target_counts[1] if 1 in target_counts else None
    }
    
    return info

def verify_notebook_environment() -> Dict:
    """
    Verifica que el entorno de notebook esté configurado correctamente.
    
    Returns:
        Dict: Estado del entorno
    """
    current_dir = Path.cwd()
    
    # Verificar estructura de proyecto
    if current_dir.name == 'notebooks':
        project_root = current_dir.parent
    else:
        project_root = current_dir
    
    verification = {
        'current_directory': str(current_dir),
        'project_root': str(project_root),
        'is_notebook_dir': current_dir.name == 'notebooks',
        'paths_exist': {
            'config': (project_root / 'config' / 'model_params.yaml').exists(),
            'src': (project_root / 'src' / 'data_loader.py').exists(),
            'data_raw': (project_root / 'data' / 'raw').exists(),
            'data_processed': (project_root / 'data' / 'processed').exists()
        }
    }
    
    # Verificar archivos de datos
    data_raw_dir = project_root / 'data' / 'raw'
    if data_raw_dir.exists():
        verification['data_files'] = {
            'train.csv': (data_raw_dir / 'train.csv').exists(),
            'test.csv': (data_raw_dir / 'test.csv').exists(),
            'demograficas.xlsx': (data_raw_dir / 'train_test_demograficas.xlsx').exists(),
            'subsidios.xlsx': (data_raw_dir / 'train_test_subsidios.xlsx').exists()
        }
    
    return verification

def print_environment_status():
    """Imprime el estado del entorno de forma legible."""
    
    verification = verify_notebook_environment()
    
    print("=" * 50)
    print("VERIFICACIÓN DEL ENTORNO DE NOTEBOOK")
    print("=" * 50)
    
    print(f"Directorio actual: {verification['current_directory']}")
    print(f"Directorio del proyecto: {verification['project_root']}")
    print(f"¿Estamos en notebooks/? {'✅' if verification['is_notebook_dir'] else '❌'}")
    
    print(f"\nARCHIVOS DEL PROYECTO:")
    for item, exists in verification['paths_exist'].items():
        status = '✅' if exists else '❌'
        print(f"  {item}: {status}")
    
    if 'data_files' in verification:
        print(f"\nARCHIVOS DE DATOS:")
        for file, exists in verification['data_files'].items():
            status = '✅' if exists else '❌'
            print(f"  {file}: {status}")
    
    # Mostrar recomendaciones si hay problemas
    missing_paths = [k for k, v in verification['paths_exist'].items() if not v]
    if missing_paths:
        print(f"\n⚠️ PROBLEMAS DETECTADOS:")
        for missing in missing_paths:
            print(f"  - Falta: {missing}")
    
    if 'data_files' in verification:
        missing_data = [k for k, v in verification['data_files'].items() if not v]
        if missing_data:
            print(f"  - Archivos de datos faltantes: {missing_data}")
    
    if not missing_paths and (not 'data_files' in verification or not missing_data):
        print(f"\n✅ ENTORNO CONFIGURADO CORRECTAMENTE")
    
    return verification