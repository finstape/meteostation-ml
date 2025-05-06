import joblib
import numpy as np
import pandas as pd

# Загрузка модели и масштабатора
model = joblib.load('rain_model_v161.pkl')
scaler = joblib.load('rain_scaler_v161.pkl')

# Пример входных данных
temperature = 4 # температура в °C
humidity = 0.305     # влажность (в долях, от 0 до 1)
pressure_mmHg = 752
pressure_mbar = pressure_mmHg * 1.33322  # перевод в миллибары
hour = 22  # текущий час

# Подготовка признаков
features = pd.DataFrame([[temperature, humidity, pressure_mbar, hour]],
                        columns=['Temperature (C)', 'Humidity', 'Pressure (millibars)', 'Hour'])

# Масштабирование и предсказание
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)[0]
prediction_text = "будет дождь ☔" if prediction == 1 else "дождя не будет 🌤"

print(prediction_text)

