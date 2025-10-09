from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)   # <- IMPORTANT: instantiate Flask with __name__

# Load your saved model (pipeline or calibrated classifier that handles preprocessing)
model = joblib.load("model/optimized_heart_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/result')
def result():
    # Read and convert inputs
    inputs = {
        'age': int(request.args.get('age')),
        'sex': int(request.args.get('sex')),
        'cp': int(request.args.get('cp')),
        'trestbps': float(request.args.get('trestbps')),
        'chol': float(request.args.get('chol')),
        'fbs': int(request.args.get('fbs')),
        'restecg': int(request.args.get('restecg')),
        'thalach': float(request.args.get('thalach')),
        'exang': int(request.args.get('exang')),
        'oldpeak': float(request.args.get('oldpeak')),
        'slope': int(request.args.get('slope')),
        'ca': int(request.args.get('ca')),
        'thal': int(request.args.get('thal'))
    }

    df = pd.DataFrame([inputs])

    # If model is a pipeline (includes preprocessing), call it directly on raw df:
    prob_high = model.predict_proba(df)[0][1]

    # Fixed risk thresholds (example)
    if prob_high < 0.2:
        risk = "Low"
        color = "green"
        message = "✅ Low risk. Keep maintaining a healthy lifestyle!"
    elif prob_high < 0.5:
        risk = "Medium"
        color = "orange"
        message = "⚠️ Medium risk. Consider lifestyle changes and consult a doctor."
    else:
        risk = "High"
        color = "red"
        message = "❌ High risk. Please consult a doctor immediately!"

    return render_template(
        "result.html",
        risk=risk,
        prob=round(prob_high * 100, 2),
        color=color,
        message=message
    )

if __name__ == "__main__":
    app.run(debug=True)