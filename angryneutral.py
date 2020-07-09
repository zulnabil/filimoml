import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from flask import Flask
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from database.db import initialize_db
from database.models import Speech
from mongoengine.queryset.visitor import Q
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

app.config['MONGODB_SETTINGS'] = {
  'host': 'mongodb://localhost/filimo-speech'
}

initialize_db(app)

speeches = Speech.objects(Q(label='male_angry') | Q(label='female_angry') | Q(label='male_neutral') | Q(label='female_neutral')).to_json()

df = pd.read_json(speeches)

mfcc_dataframe = df[['mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'label']]
mfcc_dataframe = mfcc_dataframe.rename(columns={ 'label': 'target' })
mfcc_dataframe.loc[mfcc_dataframe['target'] == 'male_angry', 'target'] = 'angry'
mfcc_dataframe.loc[mfcc_dataframe['target'] == 'female_angry', 'target'] = 'angry'
mfcc_dataframe.loc[mfcc_dataframe['target'] == 'male_neutral', 'target'] = 'neutral'
mfcc_dataframe.loc[mfcc_dataframe['target'] == 'female_neutral', 'target'] = 'neutral'

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
  layers.Dense(400, activation='relu', input_dim=13),
  layers.Dense(200, activation='relu'),
  layers.Dense(50, activation='relu'),
  layers.Dense(2, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=5000)

pred_train= model.predict(x_train)
scores = model.evaluate(x_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

pred_test= model.predict(x_test)
scores2 = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))

model.save('angryneutral_ml_model')
