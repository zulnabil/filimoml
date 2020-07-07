import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from flask import Flask
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from database.db import initialize_db
from database.models import Speech
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

app.config['MONGODB_SETTINGS'] = {
  'host': 'mongodb://localhost/filimo-speech'
}

initialize_db(app)

speeches = Speech.objects().to_json()

df = pd.read_json(speeches)
mfcc_dataframe = df[['mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'label']]
mfcc_dataframe = mfcc_dataframe.rename(columns={ 'label': 'target' })

x = mfcc_dataframe[['mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12']]
y = mfcc_dataframe[['target']]
le = preprocessing.LabelEncoder()
y = y.apply(le.fit_transform)

# print(len(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(le.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7]))

model = tf.keras.Sequential([
  layers.Dense(1000, activation='relu', input_dim=13),
  layers.Dense(500, activation='relu'),
  layers.Dense(250, activation='relu'),
  layers.Dense(75, activation='relu'),
  layers.Dense(25, activation='relu'),
  layers.Dense(8, activation='softmax'),
])

model.compile(optimizer='adamax',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=5000)

pred_train= model.predict(x_train)
scores = model.evaluate(x_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

pred_test= model.predict(x_test)
scores2 = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))

model.save('filimo_ml_model')

# oneData = Speech.objects.get(id='5eff456b2f7a579ae5c3b8f5').to_json()
# df_oneData = pd.read_json(oneData)
# df_oneData = df_oneData[['mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12']]
# print(model.predict(df_oneData))
# print(y_test)
