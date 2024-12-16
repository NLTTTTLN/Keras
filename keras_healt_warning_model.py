import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dữ liệu
health_data = pd.read_csv("health_data_with_activity_and_temp.csv")

# 2. Tiền xử lý dữ liệu

# Chuyển "Trạng Thái Di Chuyển" thành số
# Nghỉ ngơi = 0, Đi bộ = 1, Hoạt động = 2
activity_mapping = {'Nghỉ ngơi': 0, 'Đi bộ': 1, 'Hoạt động': 2}
health_data['Trạng Thái Di Chuyển'] = health_data['Trạng Thái Di Chuyển'].map(activity_mapping)

# Chuyển "Tình Trạng Sức Khỏe" thành số
# Bình thường = 0, Cảnh báo = 1, Nguy kịch = 2
health_status_mapping = {'Bình thường': 0, 'Cảnh báo': 1, 'Nguy kịch': 2}
health_data['Tình Trạng Sức Khỏe'] = health_data['Tình Trạng Sức Khỏe'].map(health_status_mapping)

# 3. Tách dữ liệu thành X (features) và y (labels)
# Các cột đầu vào: 'Nhịp Tim (bpm)', 'Nồng độ Oxy (%)', 'Nhiệt độ Cơ thể (°C)', 'Trạng Thái Di Chuyển'
X = health_data[['Nhịp Tim (bpm)', 'Nồng độ Oxy (%)', 'Nhiệt độ Cơ thể (°C)', 'Trạng Thái Di Chuyển']].values
# Cột đầu ra: 'Tình Trạng Sức Khỏe'
y = health_data['Tình Trạng Sức Khỏe'].values

# 4. Phân chia dữ liệu thành tập Train, Validation và Test
x_train_val, x_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=42)

# 5. Xây dựng Neural Network
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=4))  # input_dim = 4 vì có 4 features
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Vì có 3 lớp output: Bình thường, Cảnh báo, Nguy kịch
model.summary()

# 6. Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 7. Training model
history = model.fit(x_train, y_train, epochs=20, batch_size=10, validation_data=(x_val, y_val))

# 8. Đánh giá mô hình trên tập test
loss, accuracy = model.evaluate(x_test, y_test)
print('Test model loss:', loss)
print('Test model accuracy:', accuracy)

# 9. Save model
model.save("keras_health_warning_model.h5")

# 10. Vẽ biểu đồ thống kê

# Biểu đồ phân bố nhịp tim
plt.figure(figsize=(10, 6))
sns.histplot(health_data['Nhịp Tim (bpm)'], kde=True, color='blue', bins=20)
plt.title('Phân bố Nhịp Tim (bpm)')
plt.xlabel('Nhịp Tim (bpm)')
plt.ylabel('Tần suất')
plt.show()

# Biểu đồ phân bố nồng độ oxy
plt.figure(figsize=(10, 6))
sns.histplot(health_data['Nồng độ Oxy (%)'], kde=True, color='green', bins=20)
plt.title('Phân bố Nồng độ Oxy (%)')
plt.xlabel('Nồng độ Oxy (%)')
plt.ylabel('Tần suất')
plt.show()

# Biểu đồ phân bố nhiệt độ cơ thể
plt.figure(figsize=(10, 6))
sns.histplot(health_data['Nhiệt độ Cơ thể (°C)'], kde=True, color='red', bins=20)
plt.title('Phân bố Nhiệt độ Cơ thể (°C)')
plt.xlabel('Nhiệt độ Cơ thể (°C)')
plt.ylabel('Tần suất')
plt.show()

# Biểu đồ phân bố trạng thái di chuyển
plt.figure(figsize=(10, 6))
sns.countplot(x='Trạng Thái Di Chuyển', data=health_data, palette='Set2')
plt.title('Phân bố Trạng Thái Di Chuyển')
plt.xlabel('Trạng Thái Di Chuyển')
plt.ylabel('Số lượng')
plt.show()

# Biểu đồ phân bố tình trạng sức khỏe
plt.figure(figsize=(10, 6))
sns.countplot(x='Tình Trạng Sức Khỏe', data=health_data, palette='Set1')
plt.title('Phân bố Tình Trạng Sức Khỏe')
plt.xlabel('Tình Trạng Sức Khỏe')
plt.ylabel('Số lượng')
plt.show()

# Vẽ biểu đồ accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Vẽ biểu đồ loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
