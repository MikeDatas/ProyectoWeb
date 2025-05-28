import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import pickle
import os
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["DEBUG"] = True

print("El fichero que se está ejecutando es:")
print(__file__)



# print("... que está en el directorio:")
# print(os.path.dirname(__file__))
# os.chdir(os.path.dirname(__file__))
root_path = "/home/ProyectoWeb25/ProyectoWeb/"

# Carga el modelo UNA vez (al arrancar la app)
model_path = os.path.join(os.path.dirname(__file__), 'model_reg.pkl')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
model = pickle.load(open(model_path, 'rb'))

# End Point "/"
@app.route('/', methods=['GET'])
def home():
    # return "<h1>My API</h1><p>Ésta es una API para predicción de calidad de agua superficial.</p>"
    return send_from_directory(".", "index.html")


@app.route('/v1/predict', methods=['GET'])
def predict():
    # Obtener parámetros
    ammonia_mg_l = request.args.get('ammonia_mg_l')
    biochemical_oxygen_demand_mg_l = request.args.get('biochemical_oxygen_demand_mg_l')
    dissolved_oxygen_mg_l = request.args.get('dissolved_oxygen_mg_l')
    orthophosphate_mg_l = request.args.get('orthophosphate_mg_l')
    ph_ph_units = request.args.get('ph_ph_units')
    temperature_cel = request.args.get('temperature_cel')
    nitrogen_mg_l = request.args.get('nitrogen_mg_l')
    nitrate_mg_l = request.args.get('nitrate_mg_l')

    # Verificar que ninguno sea None (faltan parámetros)
    if None in [ammonia_mg_l, biochemical_oxygen_demand_mg_l, dissolved_oxygen_mg_l,
                orthophosphate_mg_l, ph_ph_units, temperature_cel,
                nitrogen_mg_l, nitrate_mg_l]:
        return jsonify({"error": "Faltan parámetros obligatorios"}), 400

    try:
        features = [
            float(ammonia_mg_l),
            float(biochemical_oxygen_demand_mg_l),
            float(dissolved_oxygen_mg_l),
            float(orthophosphate_mg_l),
            float(ph_ph_units),
            float(temperature_cel),
            float(nitrogen_mg_l),
            float(nitrate_mg_l),
        ]
    except ValueError:
        return jsonify({"error": "Parámetros no válidos, deben ser números"}), 400

    # Predicción
    try:
        prediction = model.predict([features])
        return jsonify({'predictions': prediction[0]})
    except Exception as e:
        return jsonify({"error": f"Error en la predicción: {str(e)}"}), 500

@app.route('/v1/retrain', methods=['GET'])
def retrain():
    if os.path.exists(root_path + "data/aguas_new.csv"):
        data = pd.read_csv(root_path +'data/aguas_new.csv')
        data.columns = [col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_") for col in data.columns]
        feature_cols = [
            'ammonia_mg_l',
            'biochemical_oxygen_demand_mg_l',
            'dissolved_oxygen_mg_l',
            'orthophosphate_mg_l',
            'ph_ph_units',
            'temperature_cel',
            'nitrogen_mg_l',
            'nitrate_mg_l'
        ]

        # 1. Datos
        X = data[feature_cols]
        y = data['ccme_values']

        X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size = 0.20,
                                                        random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        pickle.dump(model, open('model_reg.pkl', 'wb'))

        return f"Model retrained. New evaluation metric RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²:{r2:.3f} "
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
        filename = secure_filename("aguas_new.csv")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return 'File successfully uploaded and saved as aguas_new.csv'
    else:
        return 'Invalid file type. Only .csv allowed.', 400

if __name__ == "__main__":
    app.run()