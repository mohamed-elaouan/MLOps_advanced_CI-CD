import joblib
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

MODEL_PATH = "artifacts/models/model.pkl"
model = joblib.load(MODEL_PATH)


FEATURES = [
    'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
    'Temp3pm', 'RainToday', 'Year', 'Month', 'Day'
]

FIELD_METADATA = {
    "Location": ("Location code", "Example: 24", "Encoded station/location value used by the model."),
    "MinTemp": ("Minimum temperature", "Example: 13.4", "Degrees Celsius."),
    "MaxTemp": ("Maximum temperature", "Example: 22.9", "Degrees Celsius."),
    "Rainfall": ("Rainfall", "Example: 0.6", "Millimeters recorded today."),
    "Evaporation": ("Evaporation", "Example: 4.8", "Millimeters."),
    "Sunshine": ("Sunshine", "Example: 8.3", "Daily sunshine hours."),
    "WindGustDir": ("Wind gust direction code", "Example: 13", "Encoded wind direction value."),
    "WindGustSpeed": ("Wind gust speed", "Example: 44", "Kilometers per hour."),
    "WindDir9am": ("9 AM wind direction code", "Example: 13", "Encoded wind direction value."),
    "WindDir3pm": ("3 PM wind direction code", "Example: 14", "Encoded wind direction value."),
    "WindSpeed9am": ("9 AM wind speed", "Example: 20", "Kilometers per hour."),
    "WindSpeed3pm": ("3 PM wind speed", "Example: 24", "Kilometers per hour."),
    "Humidity9am": ("9 AM humidity", "Example: 71", "Relative humidity percentage."),
    "Humidity3pm": ("3 PM humidity", "Example: 22", "Relative humidity percentage."),
    "Pressure9am": ("9 AM pressure", "Example: 1007.7", "Hectopascals."),
    "Pressure3pm": ("3 PM pressure", "Example: 1007.1", "Hectopascals."),
    "Cloud9am": ("9 AM cloud cover", "Example: 8", "Cloud cover scale used by the dataset."),
    "Cloud3pm": ("3 PM cloud cover", "Example: 5", "Cloud cover scale used by the dataset."),
    "Temp9am": ("9 AM temperature", "Example: 16.9", "Degrees Celsius."),
    "Temp3pm": ("3 PM temperature", "Example: 21.8", "Degrees Celsius."),
    "RainToday": ("Rain today code", "Example: 0", "Encoded value: 0 for No, 1 for Yes."),
    "Year": ("Year", "Example: 2015", "Observation year."),
    "Month": ("Month", "Example: 6", "Observation month number."),
    "Day": ("Day", "Example: 12", "Observation day of month."),
}

FEATURE_GROUPS = [
    {
        "title": "Location and Date",
        "description": "Encoded station and observation date.",
        "features": ["Location", "Year", "Month", "Day"],
    },
    {
        "title": "Temperature and Rain",
        "description": "Core weather observations from the current day.",
        "features": ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "Temp9am", "Temp3pm", "RainToday"],
    },
    {
        "title": "Wind Conditions",
        "description": "Wind speed, gust, and encoded direction readings.",
        "features": ["WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am", "WindSpeed3pm"],
    },
    {
        "title": "Humidity, Pressure, and Cloud",
        "description": "Atmospheric signals that help classify rain risk.",
        "features": ["Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm"],
    },
]

LABELS = {0: "NO", 1: "YES"}


def build_feature_groups():
    grouped_fields = []

    for group in FEATURE_GROUPS:
        fields = []
        for feature in group["features"]:
            label, placeholder, helper = FIELD_METADATA[feature]
            fields.append(
                {
                    "name": feature,
                    "label": label,
                    "placeholder": placeholder,
                    "helper": helper,
                }
            )

        grouped_fields.append(
            {
                "title": group["title"],
                "description": group["description"],
                "fields": fields,
            }
        )

    return grouped_fields


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_values = {}

    if request.method == "POST":
        form_values = request.form.to_dict()
        try:
            input_data = [float(request.form[feature]) for feature in FEATURES]
            input_array = np.array(input_data).reshape(1, -1)

            pred = model.predict(input_array)[0]
            prediction = LABELS.get(pred, 'Unknown')
            print(prediction)

        except Exception as e:
            error = "Please enter numeric values for every field. Encoded categories such as Location, Wind Direction, and Rain Today must be submitted as numbers."
            print(str(e))

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_values=form_values,
        feature_groups=build_feature_groups(),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
