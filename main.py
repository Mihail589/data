import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model("drone_detector.keras")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Параметры аудио
CHUNK = 1024
SAMPLE_RATE = 22050  # Частота дискретизации аудио
DURATION = 2  # Длительность аудиофрагмента в секундах
N_MFCC = 13  # Количество MFCC-коэффициент

def predict_drone(audio_file):
    mfccs = extract_mfcc(audio_file)
    prediction = model.predict(np.expand_dims(mfccs, axis=0))
    if prediction[0][0] > 0.5:
        return 'Дрон'
    else:
        return 'Не дрон'

# --- Пример использования ---
audio_file = 'dron.wav'
result = predict_drone(audio_file)
print(f'Результат: {result}')