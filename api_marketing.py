import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import Lasso
import pickle
import os
from flask import Flask, jsonify, request, abort, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["DEBUG"] = True

print("El fichero que se está ejecutando es:")
print(__file__)



# print("... que está en el directorio:")
# print(os.path.dirname(__file__))
# os.chdir(os.path.dirname(__file__))
root_path = "/home/ProyectoWeb25/ProyectoWeb/"


# End Point "/"
@app.route('/', methods=['GET'])
def home():
    # return "<h1>My API</h1><p>Ésta es una API para predicción de ventas en función de inversión en marketing.</p>"
    return send_from_directory(".", "index.html")


@app.route('/v1/predict', methods=['GET'])
def predict():
    model = pickle.load(open('reg_pipeline_xgb_water.pkl','rb'))
    ammonia_mg_l = request.args.get('ammonia_mg_l', None)
    biochemical_oxygen_demand_mg_l = request.args.get('biochemical_oxygen_demand_mg_l', None)
    dissolved_oxygen_mg_l = request.args.get('dissolved_oxygen_mg_l', None)

    orthophosphate_mg_l = request.args.get('orthophosphate_mg_l', None)
    ph_ph_units = request.args.get('ph_ph_units', None)
    temperature_cel = request.args.get('temperature_cel', None)

    nitrogen_mg_l = request.args.get('nitrogen_mg_l', None)
    nitrate_mg_l = request.args.get('nitrate_mg_l', None)

    #print(ammonia_mg_l,biochemical_oxygen_demand_mg_l,dissolved_oxygen_mg_l,orthophosphate_mg_l,ph_ph_units,temperature_cel,nitrogen_mg_l,nitrate_mg_l)
    #print(type(tv))

    if ammonia_mg_l is None or biochemical_oxygen_demand_mg_l is None or dissolved_oxygen_mg_l or None or orthophosphate_mg_l is None or ph_ph_units is None or temperature_cel is None or nitrogen_mg_l is None or nitrate_mg_l is None:
        return "Args empty, the data are not enough to predict"
    else:
        prediction = model.predict([[float(ammonia_mg_l),float(biochemical_oxygen_demand_mg_l),float(dissolved_oxygen_mg_l),
                                     float(orthophosphate_mg_l),float(ph_ph_units),float(temperature_cel),
                                     float(nitrogen_mg_l),float(nitrate_mg_l)]])
    # [[float(tv),float(radio),float(newspaper)]]
    # [pred1]
    return jsonify({'predictions': prediction[0]})

@app.route('/v1/retrain', methods=['GET'])
def retrain():
    if os.path.exists(root_path + "data/Advertising_new.csv"):
        data = pd.read_csv(root_path +'data/Advertising_new.csv')

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                        data['sales'],
                                                        test_size = 0.20,
                                                        random_state=42)

        model = Lasso(alpha=6000)
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(data.drop(columns=['sales']), data['sales'])
        pickle.dump(model, open('ad_model.pkl', 'wb'))

        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

UPLOAD_FOLDER = os.path.join(root_path, 'data')
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/v1/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part in request', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        filename = secure_filename("Advertising_new.csv")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return 'File successfully uploaded and saved as Advertising_new.csv'
    else:
        return 'Invalid file type. Only .csv allowed.', 400

if __name__ == "__main__":
    app.run()