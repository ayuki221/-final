import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# 資料讀取
def load_images_and_labels(path_fox, path_non_fox, target_size=(128, 128)):
    images, labels = [], []
    for path, label in [(path_fox, 1), (path_non_fox, 0)]:
        for filename in os.listdir(path):
            if filename.endswith('.jpg'):
                img = Image.open(os.path.join(path, filename)).convert('RGB')
                img = img.resize(target_size)
                images.append(np.array(img))
                labels.append(label)
    return np.array(images), np.array(labels)

path_fox = 'C:/Users/stars/Desktop/w2_final/fox/data/train/animal fox'
path_non_fox = 'C:/Users/stars/Desktop/w2_final/fox/data/train/non fox'

images, labels = load_images_and_labels(path_fox, path_non_fox)
images = images / 255.0
labels = to_categorical(labels, num_classes=2)

# 資料分割
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 建立模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.summary()

# 資料增強
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model.fit(datagen.flow(X_train, y_train, batch_size=32), 
          validation_data=(X_test, y_test), 
          epochs=50, callbacks=[early_stopping])

# 評估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print("測試損失:", test_loss)
print("測試準確率:", test_acc)
