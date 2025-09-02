import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from glob import glob
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
 
 #dataset
UAV_PATH = UAV_PATH = r"C:\Users\tangu\Downloads\DroneAudioDataset-master\DroneAudioDataset-master\Binary_Drone_Audio\yes_drone"
NON_UAV_PATH = r"C:\Users\tangu\Downloads\DroneAudioDataset-master\DroneAudioDataset-master\Binary_Drone_Audio\unknown"


# Preprocess Audio
def preprocess_audio(file_path, sample_rate=16000):
    y, sr = librosa.load(file_path, sr=sample_rate)
    y = librosa.util.normalize(y)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return y_trimmed, sr

# Extract Spectrogram Image
def extract_spectrogram_image(y, sr):
    import matplotlib
    matplotlib.use("Agg")
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(2.27, 2.27), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    librosa.display.specshow(S_dB, sr=sr, ax=ax)
    ax.axis('off')
    canvas.draw()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    image = image.reshape((height, width, 4))
    plt.close(fig)
    return image[:, :, :3] / 255.0
#  Load Dataset

def load_dataset():
    X, y = [], []
    uav_files = glob(os.path.join(UAV_PATH, "*.wav"))[:1300]      # Balance classes
    non_uav_files = glob(os.path.join(NON_UAV_PATH, "*.wav"))[:1300]

    print(f" Scanning: {UAV_PATH} - {len(uav_files)} files")
    print(f"Scanning: {NON_UAV_PATH} - {len(non_uav_files)} files")

    for f in uav_files:
        try:
            y_audio, sr = preprocess_audio(f)
            img = extract_spectrogram_image(y_audio, sr)
            X.append(img)
            y.append(1)
        except Exception as e:
            print(f" Error loading {f}: {e}")

    for f in non_uav_files:
        try:
            y_audio, sr = preprocess_audio(f)
            img = extract_spectrogram_image(y_audio, sr)
            X.append(img)
            y.append(0)
        except Exception as e:
            print(f" Error loading {f}: {e}")

    print(f"✅ Total loaded samples: {len(X)}")
    return np.array(X), np.array(y)

# Start Training
X, y = load_dataset()
X = X.astype(np.float32)
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print("Train class distribution:", Counter(y_train))
print("Test class distribution:", Counter(y_test))

#  Class Weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print(" Class weights:", class_weights_dict)

#  Build CNN

def build_cnn(input_shape=(227, 227, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_cnn()

#  Train
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=16,
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)

#  Save Model
model.save("uav_binary_cnn_model.h5")
print("✅ Model saved as uav_binary_cnn_model.h5")

#  Evaluate
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))
print("Predictions:", np.unique(y_pred, return_counts=True))