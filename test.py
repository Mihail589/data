import tensorflow as tf
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import os
from PyQt5.QtWidgets import *
import ui  # Предполагается, что файл ui.py содержит дизайн GUI
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys

class Example(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):  # Исправлено: __init__ вместо init
        super().__init__()
        self.setupUi(self)
        self.CHUNK = 1024
        self.SAMPLE_RATE = 22050
        self.DURATION = 2
        self.N_MFCC = 13
        self.pushButton.clicked.connect(self.ner)
        self.a = 0

        # Загрузка модели TensorFlow Lite
        self.interpreter = tf.lite.Interpreter(model_path="drone_detector.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def ner(self):
        while True:
            self.record_audio("record.wav", 2, self.SAMPLE_RATE, 1)
            audio_file = 'record.wav'
            self.a += 1
            result = self.predict_drone(audio_file)
            self.label.setText(result)
            self.repaint()
            print(f"[INFO] {self.a} {result}")
            os.remove(audio_file)

    def record_audio(self, filename, duration, samplerate, channels):
        print("Начинаю запись...")
        myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
        sd.wait()
        sf.write(filename, myrecording, samplerate)

    def extract_mfcc(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.SAMPLE_RATE, duration=self.DURATION)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.N_MFCC)
        return mfccs.T

    def predict_drone(self, audio_file):
        mfccs = self.extract_mfcc(audio_file)

        # Предобработка данных для TensorFlow Lite
        input_data = np.expand_dims(mfccs, axis=0).astype(np.float32) # Проверьте тип данных на входе модели Lite!

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        if output_data[0][0] > 0.5:
            return 'Дрон'
        else:
            return 'Не дрон'

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Example()
    form.show()
    app.exec_()
