import os
import math
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras import backend as K

# 定義二元Focal Loss函式
def binary_focal_loss(gamma=2.0, alpha=0.25):

    def focal_loss(y_true, y_pred):
        # 定義epsilon以避免除以零的錯誤
        epsilon = K.epsilon()

        # 為避免log(0)，對預測值加入極小值
        y_pred = y_pred + epsilon

        # 計算Focal Loss
        focal_loss = -y_true * alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred) \
                     - (1 - y_true) * (1 - alpha) * K.pow(y_pred, gamma) * K.log(1 - y_pred)

        return K.mean(focal_loss)

    return focal_loss

# 定義學習率調整函式
def step_decay(epoch):
    initial_lrate = 0.01  # 初始學習率
    drop = 0.5  # 學習率下降比例
    epochs_drop = 10.0  # 每多少個epoch降低學習率
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

# 設定資料集路徑
path_fox = 'C:/Users/stars/Desktop/w2_final/fox/data/train/animal fox'
path_non_fox = 'C:/Users/stars/Desktop/w2_final/fox/data/train/non fox'

# 建立儲存影像和標籤的列表
images = []
labels = []
target_size = (128, 128)
# 迴圈讀取資料夾中的影像
for filename in os.listdir(path_fox):
    if filename.endswith('.jpg'):
        img = Image.open(path_fox + '/' + filename).convert('RGB')  # 確保為RGB格式
        img = img.resize(target_size)  # 統一尺寸
        images.append(np.array(img))
        labels.append(1)

for filename in os.listdir(path_non_fox):
    if filename.endswith('.jpg'):
        img = Image.open(path_non_fox + '/' + filename).convert('RGB')  # 確保為RGB格式
        img = img.resize(target_size)  # 統一尺寸
        images.append(np.array(img))
        labels.append(0)

# 將影像與標籤轉為numpy陣列
images = np.array(images)
labels = np.array(labels)

# 正規化影像（將像素值縮放至[0,1]）
images = images / 255.0

# 將標籤進行One-hot編碼
labels = to_categorical(labels, num_classes=2)

# 取得第一張影像的形狀作為輸入層的input_shape
img_shape = images[0].shape

# 將資料集分為訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 建立模型
model = Sequential([
    Input(shape=img_shape),  # 使用Input層指定輸入形狀
    Conv2D(32, kernel_size=(3, 3), activation='relu'),  # 卷積層
    MaxPooling2D(pool_size=(2, 2)),  # 最大池化層
    Dropout(0.25),  # Dropout層
    Conv2D(64, kernel_size=(3, 3), activation='relu'),  # 增加第二個卷積層
    MaxPooling2D(pool_size=(2, 2)),  # 最大池化層
    Dropout(0.25),  # Dropout層
    Conv2D(128, kernel_size=(3, 3), activation='relu'),  # 增加第三個卷積層
    MaxPooling2D(pool_size=(2, 2)),  # 最大池化層
    Dropout(0.25),  # Dropout層
    Flatten(),  # 展平層
    Dense(128, activation='relu'),  # 全連接層
    Dropout(0.5),  # Dropout層
    Dense(2, activation='softmax')  # 輸出層（2類分類）
])

# 編譯模型，使用二元Focal Loss作為損失函數
model.compile(optimizer='adam', loss=binary_focal_loss(), metrics=['accuracy'])

# 使用資料增強技術來增加訓練資料量
datagen = ImageDataGenerator(
    rotation_range=30,  # 旋轉範圍
    width_shift_range=0.2,  # 水平平移範圍
    height_shift_range=0.2,  # 垂直平移範圍
    zoom_range=0.2,  # 縮放範圍
    horizontal_flip=True,  # 隨機水平翻轉
    vertical_flip=True,  # 隨機垂直翻轉
    fill_mode='nearest')  # 填充模式

datagen.fit(images)  # 對影像進行擴增

# 使用Early Stopping以防止過度擬合
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# 設定測試資料集路徑
test_path_fox = 'C:/Users/stars/Desktop/w2_final/fox/data/test/animal fox'
test_path_non_fox = 'C:/Users/stars/Desktop/w2_final/fox/data/test/non fox'

# 建立儲存測試影像與標籤的列表
test_images = []
test_labels = []

# 迴圈讀取測試資料夾中的影像
for filename in os.listdir(test_path_fox):
    if filename.endswith('.jpg'):
        img = Image.open(test_path_fox + '/' + filename).convert('RGB')  # 確保為RGB格式
        img = img.resize(target_size)  # 統一尺寸
        test_images.append(np.array(img))
        test_labels.append(1)

for filename in os.listdir(test_path_non_fox):
    if filename.endswith('.jpg'):
        img = Image.open(test_path_non_fox + '/' + filename).convert('RGB')  # 確保為RGB格式
        img = img.resize(target_size)  # 統一尺寸
        test_images.append(np.array(img))
        test_labels.append(0)

# 將測試影像與標籤轉為numpy陣列
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# 正規化測試影像
test_images = test_images / 255.0

# 將測試標籤進行One-hot編碼
test_labels = to_categorical(test_labels, num_classes=2)

# 在測試資料上評估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("\n測試損失:", test_loss)
print("測試準確率:", test_acc)

# 使用model.predict進行預測並顯示其形狀
print("\n生成預測結果")
prediction = model.predict(test_images[:1])
print("預測結果:", prediction)
print("預測結果形狀:", prediction.shape)
