import joblib
import pandas as pd

# Загрузка модели и scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Пример входных данных
temperature = 12.5  # °C
humidity = 85.0  # %
pressure_mmhg = 752.0  # мм рт. ст.
timestamp = "2024-05-18 14:00"  # строка в формате YYYY-MM-DD HH:MM

# Преобразование даты
dt = pd.to_datetime(timestamp)
hour = dt.hour
day = dt.day
month = dt.month

# Формирование признаков
X_input = pd.DataFrame([{
    'Temperature': temperature,
    'Humidity': humidity,
    'Pressure (mmHg)': pressure_mmhg,
    'Month': month,
    'Hour': hour,
    'DayOfMonth': day
}])

# Масштабирование
X_scaled = scaler.transform(X_input)

# Предсказание
y_pred = model.predict(X_scaled)[0]
predicted_temp = y_pred[0]
predicted_rain = int(y_pred[1] >= 0.5)

# Вывод
print(f"Температура через 6 часов: {predicted_temp:.1f} °C")
print(f"Будет ли дождь: {'Да' if predicted_rain else 'Нет'}")
