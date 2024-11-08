from tensorflow.keras.models import load_model
import numpy as np
import librosa

# Fungsi untuk load dan ekstrak Mel-spectrogram
def load_and_extract_spectrogram(file_path, n_mels=128, n_fft=2048, hop_length=512, max_time_steps=128):
    y, sr = librosa.load(file_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec_db.shape[1] < max_time_steps:
        pad_width = max_time_steps - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :max_time_steps]
    return mel_spec_db

# Fungsi untuk melakukan prediksi
def predict_speech_class(model_path, audio_file):
    # Muat model yang sudah dilatih
    model = load_model(model_path)
    
    # Ekstraksi fitur dari file audio yang ingin diuji
    spectrogram = load_and_extract_spectrogram(audio_file)
    
    # Tambahkan dimensi ekstra agar sesuai dengan input model
    spectrogram = spectrogram[np.newaxis, ..., np.newaxis]  # Tambah batch dimension dan channel

    # Prediksi menggunakan model yang dimuat
    prediction = model.predict(spectrogram)
    print(prediction)
    
    # Hasil prediksi (misalnya: 0 atau 1 untuk klasifikasi biner)
    prediction_label = int(np.round(prediction[0][0]))  # Membulatkan prediksi ke 0 atau 1
    confidence = prediction[0][0] * 100  # Konversi confidence menjadi persentase

    if prediction_label == 1:
        print(f"The predicted class is: Correct pronunciation with confidence {confidence:.2f}%")
    else:
        print(f"The predicted class is: Incorrect pronunciation with confidence {confidence:.2f}%")

# Contoh pemakaian
model_path = 'model/model_01. alif_fathah.keras'  # Nama file model untuk kelas A
audio_file = 'a.wav'  # File audio yang akan diuji
predict_speech_class(model_path, "a.wav")
# predict_speech_class(model_path, "aa.wav")
# predict_speech_class(model_path, "aaa.wav")
# predict_speech_class(model_path, "aaaa.wav")
# predict_speech_class(model_path, "aaaaa.wav")
# predict_speech_class(model_path, "aaaaaa.wav")
# predict_speech_class(model_path, "aaaaaaa.wav")
# predict_speech_class(model_path, "aaaaaaaa.wav")
# predict_speech_class(model_path, "s.wav")
# predict_speech_class(model_path, "ss.wav")
# predict_speech_class(model_path, "sss.wav")
# predict_speech_class(model_path, "d.wav")
# predict_speech_class(model_path, "dd.wav")
# predict_speech_class(model_path, "ddd.wav")
# predict_speech_class(model_path, "ta.wav")