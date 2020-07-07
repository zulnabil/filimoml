from flask import Flask, request, Response
from database.db import initialize_db
from database.models import Speech
from flask_cors import CORS, cross_origin
from tensorflow import keras
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

app.config['MONGODB_SETTINGS'] = {
  'host': 'mongodb://localhost/filimo-speech'
}
cors = CORS(app)

initialize_db(app)

@app.route('/speeches')
def get_speeches():
  speeches = Speech.objects().to_json()
  return Response(speeches, mimetype="application/json", status=200)

@app.route('/speeches/<label>')
def get_speeches_with_label(label):
  speeches = Speech.objects(label=label).to_json()
  return Response(speeches, mimetype="application/json", status=200)

@app.route('/speeches/add', methods=['POST'])
def add_speech():
  body = request.get_json()
  speech = Speech(**body).save()
  id = speech.id
  return { "id": str(id) }, 200

@app.route('/speeches/predict', methods=['POST'])
@cross_origin()
def predict_speech():
  body = request.get_json()
  dataframe = pd.DataFrame(body, index=[0])
  model = keras.models.load_model("./filimo_ml_model")
  output = model.predict(dataframe).tolist()
  return {
    "status": 200,
    "output": output
  }, 200

app.run(ssl_context=('/etc/letsencrypt/live/filimoml.com/cert.pem', '/etc/letsencrypt/live/filimoml.com/privkey.pem'))
