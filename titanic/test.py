import joblib

# Load the pre-trained model
with open('model_saved', 'rb') as model_file:
    model = joblib.load(model_file)

prediction = model.predict([[1, 0, 27, 7]])
