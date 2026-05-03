import pandas as pd

# Rutas de tus archivos pesados
file_train = "data_airbnb.csv"
file_test = "data_airbnb_test.csv"

# Creamos muestras de 1000 filas (pesarán menos de 1MB)
try:
    df_train_sample = pd.read_csv(file_train, nrows=1000)
    df_test_sample = pd.read_csv(file_test, nrows=1000)

    # Guardamos los nuevos archivos (sobrescribiendo los locales para la subida)
    # NOTA: Asegúrate de tener un respaldo de tus archivos originales de 450MB en otra carpeta
    df_train_sample.to_csv("data_airbnb_sample.csv", index=False)
    df_test_sample.to_csv("data_airbnb_test_sample.csv", index=False)
    
    print("✅ Muestras creadas con éxito: data_airbnb_sample.csv y data_airbnb_test_sample.csv")
except Exception as e:
    print(f"❌ Error: {e}")