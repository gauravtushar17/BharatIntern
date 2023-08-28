from flask import Flask, render_template, request, url_for
import joblib 
import numpy as np

model = joblib.load(open('model_saved','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def home():
    Pclass = int(request.form['Pclass'])
    Sex = int(request.form['Sex'])
    Age = float(request.form['Age'])
    Fare = float(request.form['Fare'])
    prediction = model.predict([[Pclass, Sex, Age, Fare]])[0]

    return render_template('later.html', data=prediction)

if __name__ == '__main__':
    app.run(debug=True)