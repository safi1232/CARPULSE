import pickle
import pandas as pd

with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

MODEL_VERSION="1.0.0"

def predict_output(user_data):

    df = pd.DataFrame([user_data])

    print("Input columns:")
    print(df.columns.tolist())

    prediction = model.predict(df)

    print("Raw prediction:")
    print(prediction)

    return float(prediction[0])