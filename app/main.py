
from flask import Flask
import pickle
import numpy as np
from flask import render_template, request

app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    occupancy = float(request.form['occupancy'])
    queue = float(request.form['queue'])
    special = float(request.form['special'])
    feature = np.array([[occupancy, queue, special,]])
    if occupancy < 0 or occupancy > 100:
        return "Occupancy must be between 0 and 100"

    if queue < 0:
        return "Queue cannot be negative"

    if special not in [0, 1]:
        return "Special day must be 0 or 1"

    prediction = model.predict(feature)
    return render_template('index.html', prediction_text=f"₹{round(prediction[0], 2)}")


if __name__ == '__main__':
    app.run(debug=True)
