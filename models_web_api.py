# Dependencies
import os
from os import system
from flask import Flask, request, jsonify

import traceback
import pandas as pd
import numpy as np
from prediction_api import *

# Your API definition
app = Flask(__name__)

@app.route("/")
def home_view():
        return "<h1>Welcome</h1>"
    
@app.route('/predict', methods=['POST'])
def predict():
    randomForest = pickle.load(open("classifier_rf_model.sav", 'rb'))
    if randomForest:
        try:
            json_ = request.json
            print(json_)
            query = pd.DataFrame(json_)           
            X_transformed = preprocessing(query)
            y_pred = randomForest.predict(X_transformed)
            y_proba = randomForest.predict_proba(X_transformed)
            
            return jsonify({'prediction': y_pred,'prediction_proba':y_proba[0][0]})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Problem loading the model')
        return ('No model here to use')

@app.route('/predictByClientId', methods=['POST'])
def predictByClientId():
    randomForest = pickle.load(open("classifier_rf_model.sav", 'rb'))
    if randomForest:
        try:
            json_ = request.json
            print(json_)
            sample_size = 10000
            
            print(json_)  

            sample_size= 20000
            #data_set = data = pd.read_csv("df_final.csv",nrows=sample_size)
            #client=data_set[data_set['SK_ID_CURR']==json_['SK_ID_CURR']].drop(['SK_ID_CURR','TARGET'],axis=1)
            #print(client)
            
            data_set = pd.read_csv("app_test.csv", nrows=sample_size)
            client = data_set[data_set['SK_ID_CURR'] == json_['SK_ID_CURR']].drop(['SK_ID_CURR'],axis=1)
            print(client)

            
            preproc = pickle.load(open("preprocessor.sav", 'rb'))
            X_transformed =preproc.transform(client)
            y_pred = randomForest.predict(X_transformed)
            y_proba = randomForest.predict_proba(X_transformed)
            
            return jsonify({'prediction': str(y_pred[0]),'prediction_proba':str(y_proba[0][0])})


        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Problem loading the model')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
      
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345


    app.run(host="localhost", port=port, debug=False)

    
