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
root_path = "/home/ItsazainBilbao2/TallerDespliegue2503/"


# End Point "/"
@app.route('/', methods=['GET'])
def home():
    # return "<h1>My API</h1><p>Ésta es una API para predicción de ventas en función de inversión en marketing.</p>"
    return send_from_directory(".", "index.html")


@app.route('/v1/predict', methods=['GET'])
def predict():
    model = pickle.load(open('ad_model.pkl','rb'))
    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    print(tv,radio,newspaper)
    print(type(tv))

    if tv is None or radio is None or newspaper is None:
        return "Args empty, the data are not enough to predict"
    else:
        prediction = model.predict([[float(tv),float(radio),float(newspaper)]])
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