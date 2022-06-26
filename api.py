
from flask import Flask ,request,jsonify
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/Predicted',methods=['POST','GET'])
def predicted():
    pregnancies = float(request.args['pregnancies'])
    glucose = float(request.args['glucose'])
    bp = float(request.args['bp'])
    skin_thickness = float(request.args['skin_thickness'])
    insulin = float(request.args['insulin'])
    bmi = float(request.args['bmi'])
    dpf = float(request.args['dpf'])
    age = float(request.args['age'])

    X = np.array([[pregnancies, glucose, bp, skin_thickness, insulin,
                   bmi, dpf, age]])

    model_path = 'random_forest_grid.sav'
    model = joblib.load(model_path)

    Y_pred = model.predict(X)

    return str(Y_pred);

if __name__=="__main__":
    app.run()