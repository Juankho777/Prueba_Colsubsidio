# === Nuevo alias para compatibilidad con notebooks antiguos ===
def get_data_for_notebook():
    """
    Alias de load_data_for_notebook para compatibilidad.
    Devuelve un diccionario con todos los datasets cargados e integrados.
    """
    try:
        train_integrated, test_integrated = load_data_for_notebook()
        return {
            'train_integrated': train_integrated,
            'test_integrated': test_integrated
        }
    except Exception as e:
        logger.warning(f"Error usando DataLoader: {e}")
        logger.info("Usando carga alternativa...")

        data_path = Path("data/raw")
        datasets = {}

        try:
            datasets['train_integrated'] = pd.read_excel(data_path / "train.xlsx")
            datasets['test_integrated'] = pd.read_excel(data_path / "test.xlsx")
            logger.info("Archivos principales cargados")

            if (data_path / "train_test_demograficas.xlsx").exists():
                datasets['demograficas'] = pd.read_excel(data_path / "train_test_demograficas.xlsx")
                datasets['train_integrated'] = datasets['train_integrated'].merge(datasets['demograficas'], on='id', how='left')
                datasets['test_integrated'] = datasets['test_integrated'].merge(datasets['demograficas'], on='id', how='left')

            if (data_path / "train_test_subsidios.xlsx").exists():
                datasets['subsidios'] = pd.read_excel(data_path / "train_test_subsidios.xlsx")
                datasets['train_integrated'] = datasets['train_integrated'].merge(datasets['subsidios'], on='id', how='left')
                datasets['test_integrated'] = datasets['test_integrated'].merge(datasets['subsidios'], on='id', how='left')

            logger.info("Carga alternativa completada")

        except FileNotFoundError as fnf:
            logger.error(f"Archivo no encontrado: {fnf}")

        # Mostrar resumen
        print("\n=== DATASETS CARGADOS ===")
        for name, df in datasets.items():
            if isinstance(df, pd.DataFrame):
                print(f"{name.upper()}: {len(df):,} registros x {len(df.columns)} columnas")

        return datasets

