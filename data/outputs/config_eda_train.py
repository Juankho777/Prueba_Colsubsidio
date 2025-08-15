# Configuración EDA Train Colsubsidio
# Variables creadas y configuración para carga posterior

DATASET_PATH = 'train_procesado_eda.csv'
TARGET_VARIABLE = 'Target'
VARIABLES_NUEVAS = ['utilizacion_cupo', 'ratio_pago', 'cambio_saldo', 'perfil_riesgo']
VAR_MAS_PREDICTIVA = 'utilizacion_cupo'
CORRELACION_MAX = 0.078
TOTAL_CLIENTES = 50001
TASA_FUGA = 0.028
