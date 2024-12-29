# Загрузка модели TFLite
interpreter = tf.lite.Interpreter(model_path="drone_detector.tflite")
interpreter.allocate_tensors()

# Получение входных и выходных тензоров
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


from PyQt5.QtWidgets import *
import ui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
class Example(QMainWindow, ui.Ui_MainWindow):
    def __init__(self): # Исправлено: __init__ вместо init
        super().__init__()
        self.setupUi(self)
        self.CHUNK = 1024
        self.SAMPLE_RATE = 22050  # Частота дискретизации аудио
        self.DURATION = 2  # Длительность аудиофрагмента в секундах
        self.N_MFCC = 13  # Количество MFCC-коэффициент
        self.pushButton.clicked.connect(self.ner)
        self.a = 0

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
    def record_audio(self,filename, duration, samplerate, channels):
        """Записывает аудио в WAV-файл.

        Args:
            filename: Имя файла для сохранения записи (по умолчанию "recording.wav").
            duration: Длительность записи в секундах (по умолчанию 5 секунд).
            samplerate: Частота дискретизации (по умолчанию 44100 Гц).
            channels: Количество каналов (1 - моно, 2 - стерео, по умолчанию 1).
        """
        print("Начинаю запись...")
        myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
        sd.wait()  # Wait until recording is finished
        sf.write(filename, myrecording, samplerate)

    def extract_mfcc(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.SAMPLE_RATE, duration=self.DURATION)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.N_MFCC)
        return mfccs.T

    def predict_drone(self, audio_file):
        mfccs = self.extract_mfcc(audio_file)
        input_data = np.expand_dims(mfccs, axis=0).astype(np.float32) # Добавлено приведение типа

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        if output_data[0][0] > 0.5:
            return 'Дрон'
        else:
            return 'Не дрон'

if __name__ == '__main__': # Исправлено: __name__ == '__main__'
    app = QApplication(sys.argv)
    form = Example()
    form.show()
    app.exec_()