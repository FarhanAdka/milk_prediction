from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model Decision Tree
model = joblib.load('model_log_regression.pkl')

# Label untuk kelas target
target_names = ['high', 'medium', 'low']  # Sesuaikan dengan label target pada model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input dari form untuk semua fitur
        ph = float(request.form['ph'])
        temperature = float(request.form['temperature'])
        taste = float(request.form['taste'])
        odor = float(request.form['odor'])
        fat = float(request.form['fat'])
        turbidity = float(request.form['turbidity'])
        colour = float(request.form['colour'])

        # Format input ke dalam array sesuai fitur model
        features = np.array([[ph, temperature, taste, odor, fat, turbidity, colour]])

        # Prediksi menggunakan model
        prediction = model.predict(features)[0]
        prediction_label = target_names[prediction]

        # Tampilkan hasil prediksi
        return f"<h1>Prediction: {prediction_label}</h1>"

    except Exception as e:
        return f"<h1>Error: {str(e)}</h1>"

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, port=5001)
