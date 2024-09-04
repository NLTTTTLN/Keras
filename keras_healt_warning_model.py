import pandas
import numpy
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# 1. Load dữ liệu và phân mảnh dữ liệu cho Train, Val và Test
health_data = pandas.read_csv("health_data_with_activity.csv")

x = health_data[:,0:499]
y = health_data[:,499]

x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size = 0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size = 0.8)


# 2. Xây dựng Neural Network
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=5))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 3. Compile model 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Training model
model.fit(x_train, y_train, epochs=20, batch_size=10)
 
 # 5.Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)
print('Test model loss:', loss)
print('Test model accuracy:', accuracy)

# 6. Save model
model.save("keras_healt_warning_model.h5")