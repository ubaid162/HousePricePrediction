import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this in production

# Load model and encoders
model = joblib.load('model/rf_model.pkl')
encoders = joblib.load('model/encoders.pkl')  # Dictionary: 'area' -> OrdinalEncoder, 'location' -> LabelEncoder
oe = encoders['area']
le = encoders['location']

# Dummy credentials (customize for your own)
VALID_USERNAME = 'admin'
VALID_PASSWORD = 'admin123'

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# About route
@app.route('/about')
def about():
    return render_template('about.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session['user'] = username
            return redirect(url_for('index'))
        else:
            error = 'Invalid credentials. Please try again.'
    return render_template('login.html', error=error)

# Logout route
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# Prediction logic
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
    try:
        area = request.form['area']
        price_per_sqft = float(request.form['price_per_sqft'])
        location = request.form['location']
        total_sqft = float(request.form['total_sqft'])
        bath = int(request.form['bath'])
        bhk = int(request.form['bhk'])

        # Encoding
        area_encoded = oe.transform([[area]])[0][0]
        location_encoded = le.transform([location])[0]

        # Predict
        input_features = np.array([[area_encoded, price_per_sqft, location_encoded, total_sqft, bath, bhk]])
        prediction = model.predict(input_features)[0]

        return render_template('result.html', prediction=round(prediction, 2))
    except Exception as e:
        return f"❌ Error: {e}"

# Index route (protected)
@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    areas = list(oe.categories_[0])
    locations = list(le.classes_)
    return render_template('index.html', areas=areas, locations=locations)

if __name__ == '__main__':
    app.run(debug=True)
