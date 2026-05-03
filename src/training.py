import pandas as pd
import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

data_airbnb = "https://github.com/josemotta84/Proyectofinal/data_airbnb_sample.csv"

data_airbnb = pd.read_csv(data_airbnb)

# Filtramos solo las columnas numéricas para evitar errores
numeric_df = data_airbnb.select_dtypes(include=['float64', 'int64'])

# 1. Scatter Plot: Capacidad vs Precio
# Esto confirmará si la relación es lineal
plt.figure(figsize=(10, 6))
sns.regplot(data=data_airbnb, x='accommodates', y='price', scatter_kws={'alpha':0.3})
plt.title('¿A mayor capacidad, mayor precio?')
plt.show()

# 2. Boxplot: Tipo de Habitación (Feature Categórica clave)
# Aunque no sale en la matriz por ser texto, 'room_type' es vital
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_airbnb, x='room_type', y='price')
plt.yscale('log') # Escala logarítmica para ver mejor la distribución
plt.title('Impacto del Tipo de Habitación en el Precio')
plt.show()

# Convierte 'room_type' en columnas de 0 y 1
data_airbnb = pd.get_dummies(data_airbnb, columns=['room_type'], drop_first=True)

# Basado en la matriz de correlación y los gráficos:
features = ['accommodates', 'bedrooms', 'bathrooms', 'latitude', 'longitude', 
            'room_type_Private room', 'room_type_Shared room', 'room_type_Hotel room']

X = data_airbnb[features]
y = data_airbnb['price']

# Dividir: 80% entrenamiento, 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar (importante para que la latitud/longitud no dominen sobre el número de cuartos)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

features_num = ['accommodates', 'bedrooms', 'bathrooms', 'latitude', 'longitude', 'review_scores_location']
target = 'price'

# Creamos una copia para no alterar el dataframe original
df_model = data_airbnb[features_num + ['room_type', target]].copy()

# Manejo de nulos rápido para el modelo
df_model = df_model.dropna()

# 2. Codificación de variables categóricas (One-Hot Encoding)
# Esto convierte 'room_type' en columnas numéricas[cite: 1]
df_model = pd.get_dummies(df_model, columns=['room_type'], drop_first=True)

# 3. Definir X e y
X = df_model.drop(columns=[target])
y = df_model[target]

# 4. División en entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entrenar el Random Forest
print("Entrenando el modelo... esto puede tardar unos segundos.")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 6. Evaluación básica
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Resultados del Modelo ---")
print(f"Error Absoluto Medio (MAE): ${mae:.2f}")
print(f"Coeficiente de determinación (R2): {r2:.2f}")

joblib.dump(rf_model, 'rf_model_airbnb.joblib')
joblib.dump(scaler, 'scaler_airbnb.joblib')
print("Modelo y escalador guardados correctamente.")

# 7. Visualización de la Importancia de las Características
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Importancia de las Variables en la Predicción del Precio')
plt.show()

