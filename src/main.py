import joblib
import pandas as pd
from datetime import datetime

# === 1. Загрузка моделей (каждая — полный Pipeline: scaler + модель) ===
rain_model = joblib.load('best_rain_model_limited.pkl')   # дождь через 6 ч
temp_model = joblib.load('best_temp_model_limited.pkl')   # температура через 6 ч

# === 2. Пример исходных данных ===
temperature_c = 4.0        # °C
humidity_frac = 0.305      # относительная влажность (0–1)
pressure_mmHg = 752        # мм рт. ст.
ts = datetime(2025, 5, 14, 22, 0)   # текущий момент (можно заменить на now())

# === 3. Фичер-инжениринг в точности как при обучении ===
humidity_pct = humidity_frac * 100
hour  = ts.hour            # 0–23
dow   = ts.weekday()       # 0=Пн … 6=Вс
month = ts.month           # 1–12

# Формируем датафрейм-строку ровно в том же порядке столбцов,
# что использовался при обучении моделей
features = pd.DataFrame(
    [[temperature_c, humidity_pct, pressure_mmHg, hour, dow, month]],
    columns=['Temperature (C)', 'HumidityPct', 'Pressure_mmHg', 'hour', 'dow', 'month']
)

# === 4. Предсказания ===
rain_pred = rain_model.predict(features)[0]      # 0 / 1
temp_pred = temp_model.predict(features)[0]      # °C

rain_text = "будет дождь ☔" if rain_pred == 1 else "дождя не будет 🌤"

print("Прогноз осадков через 6 часов :", rain_text)
print(f"Прогноз температуры через 6 часов: {temp_pred:.1f} °C")

