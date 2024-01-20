import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

dataset = pd.read_csv("./cancer case/cancer.csv")
# C:\Users\Dan Nguyen\Desktop\AI Projects\cancer case\cancer.csv
# set X
x = dataset.drop(columns =['diagnosis(1=m, 0=b)'])

# set Y
y = dataset ["diagnosis(1=m, 0=b)"]

# input x (independent variables)
# output y (dependent variables)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2)

# initialize a sequential model
model = tf.keras.models.Sequential()

# create layers 
model.add(tf.keras.layers.InputLayer(input_shape= (x_train.shape[1],)))

model.add(tf.keras.layers.Dense(256, activation="sigmoid"))
model.add(tf.keras.layers.Dense(256, activation = "sigmoid"))
model.add(tf.keras.layers.Dense(1, activation = "sigmoid"))

model.compile(optimizer = "adam", loss = "binary_crossentropy" , metrics = ["accuracy"])
model.fit(x_train, y_train, epochs = 1000, batch_size =32)

model.evaluate(x_test, y_test)





