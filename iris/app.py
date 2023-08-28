from flask import Flask, render_template, request, url_for
import joblib 
import numpy as np

model = joblib.load(open('model_saved2', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def home():
    SepalLengthCm = float(request.form['SepalLengthCm'])
    SepalWidthCm = float(request.form['SepalWidthCm'])
    PetalLengthCm = float(request.form['PetalLengthCm'])
    PetalWidthCm = float(request.form['PetalWidthCm'])
    pred = model.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])[0]
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)