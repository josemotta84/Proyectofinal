# Airbnb Price Prediction - Taller 4

Este proyecto implementa un pipeline de Machine Learning para predecir los precios de alquiler de Airbnb utilizando un modelo de **Random Forest Regressor**. La estructura está diseñada para ser **modular, automatizada y compatible con flujos de trabajo de MLflow y GitHub Actions**.

---

## Descripción del Proyecto

El objetivo principal es identificar las características que más influyen en el precio de una propiedad (como la ubicación, capacidad y tipo de habitación) y construir un modelo capaz de estimar el costo por noche con precisión.

### Características Principales (Features)

- **Capacidad (`accommodates`)**: Número de huéspedes permitidos  
- **Habitaciones (`bedrooms`, `bathrooms`)**: Distribución física de la propiedad  
- **Ubicación (`latitude`, `longitude`)**: Coordenadas geográficas para análisis espacial  
- **Tipo de Habitación (`room_type`)**: Categorización (casa completa, habitación privada, etc.)  

---

Debido a limitaciones de git me toco crear un sampling de los datos debido a que el dataset original pesaba mas de 100 mb. Sin embargo, aca dejo los links al dataset completo

data_airbnb https://universidadeaneduco-my.sharepoint.com/:x:/g/personal/jmottam71981_universidadean_edu_co/IQCTUy5DSeCCTZgwgauy06TIATrhw1qwxRqRhUpNF49KtNU?e=wh5ike 

data_airbnb_test https://universidadeaneduco-my.sharepoint.com/:x:/g/personal/jmottam71981_universidadean_edu_co/IQDN1CkmbjbzRqPyBpFvcLJoAbBRsnagZEMAMGDqm_p8HbY?e=e7t1FO 