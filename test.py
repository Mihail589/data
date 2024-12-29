import numpy as np
import tensorflow as tf
import librosa
import tflite_runtime.interpreter as tflite

CHUNK = 1024
SAMPLE_RATE = 22050  # Частота дискретизации аудио
DURATION = 2  # Длительность аудиофрагмента в секундах
N_MFCC = 13  # Количество MFCC-коэффициент
# Загрузка интерпретатора TFLite
interpreter = tflite.Interpreter(model_path="drone_detector.tflite")
interpreter.allocate_tensors()

# Получение информации о тензорах
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def extract_mfcc(file_path):
     audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
     return mfccs.T

def predict_drone(mfccs):
    # Подготовка входных данных
    input_data = np.expand_dims(mfccs, axis=0).astype(np.float32) # Убедитесь в типе данных!

    # Запись входных данных в интерпретатор
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Запуск интерпретатора
    interpreter.invoke()

    # Получение результатов
    prediction = interpreter.get_tensor(output_details[0]['index'])

    if prediction[0][0] > 0.5:
        return 'Дрон'
    else:
        return 'Не дрон'

# --- Пример использования ---
#  (Функция extract_mfcc остается без изменений)
audio_file = 'dron.wav'
mfccs = extract_mfcc(audio_file) #  Вам нужно определить эту функцию
result = predict_drone(mfccs)
print(f'Результат: {result}')
