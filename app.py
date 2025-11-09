from flask import Flask, request, jsonify, send_from_directory, render_template, session, redirect, url_for, flash
from flask_cors import CORS
import joblib
import os
import json
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, static_folder='', template_folder='')
app.secret_key = 'your-secret-key-change-in-production'
CORS(app)

# Simple user storage (in production, use a proper database)
USERS_FILE = 'users.json'
HISTORY_FILE = 'history.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def save_prediction(username, data, result, probabilities):
    from datetime import datetime
    history = load_history()
    if username not in history:
        history[username] = []
    
    prediction_record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_data': data,
        'result': result,
        'probabilities': probabilities
    }
    
    history[username].append(prediction_record)
    save_history(history)

# Load the trained model
model = joblib.load('loan_approval_rf_model.pkl')

# Load label encoders (assuming we save them too, but for simplicity, hardcode or recreate)
# In production, save encoders as well
education_map = {'Graduate': 0, 'Not Graduate': 1}
self_employed_map = {'No': 0, 'Yes': 1}

@app.route('/', methods=['GET'])
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return send_from_directory('', 'index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        
        if username in users and check_password_hash(users[username]['password'], password):
            session['user'] = username
            return redirect(url_for('home'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()

        if username in users:
            flash('Username already exists')
        else:
            users[username] = {'password': generate_password_hash(password)}
            save_users(users)
            # Initialize empty history for new user
            history = load_history()
            history[username] = []
            save_history(history)
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/api', methods=['GET'])
def api_status():
    return jsonify({'message': 'Loan Approval Prediction API is running. Use POST /predict with JSON data.'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        # Expected input: dict with keys: no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value
        # Note: loan_id not needed for prediction

        # Preprocess
        features = [
            int(data['no_of_dependents']),
            education_map[data['education']],
            self_employed_map[data['self_employed']],
            float(data['income_annum']),
            float(data['loan_amount']),
            int(data['loan_term']),
            int(data['cibil_score']),
            float(data['residential_assets_value']),
            float(data['commercial_assets_value']),
            float(data['luxury_assets_value']),
            float(data['bank_asset_value'])
        ]

        # Predict
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        result = 'Approved' if prediction == 0 else 'Rejected'
        prob_approved = float(probabilities[0])
        prob_rejected = float(probabilities[1])

        # Save prediction to history
        save_prediction(session['user'], data, result, {
            'Approved': prob_approved,
            'Rejected': prob_rejected
        })
        
        return jsonify({
            'loan_status': result,
            'probabilities': {
                'Approved': prob_approved,
                'Rejected': prob_rejected
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/submit', methods=['POST'])
def submit():
    if 'user' not in session:
        return redirect(url_for('login'))
    try:
        data = request.form.to_dict()
        # Convert to appropriate types
        data['no_of_dependents'] = int(data['no_of_dependents'])
        data['income_annum'] = float(data['income_annum'])
        data['loan_amount'] = float(data['loan_amount'])
        data['loan_term'] = int(data['loan_term'])
        data['cibil_score'] = int(data['cibil_score'])
        data['residential_assets_value'] = float(data['residential_assets_value'])
        data['commercial_assets_value'] = float(data['commercial_assets_value'])
        data['luxury_assets_value'] = float(data['luxury_assets_value'])
        data['bank_asset_value'] = float(data['bank_asset_value'])
        
        features = [
            data['no_of_dependents'],
            education_map[data['education']],
            self_employed_map[data['self_employed']],
            data['income_annum'],
            data['loan_amount'],
            data['loan_term'],
            data['cibil_score'],
            data['residential_assets_value'],
            data['commercial_assets_value'],
            data['luxury_assets_value'],
            data['bank_asset_value']
        ]

        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        result = 'Approved' if prediction == 0 else 'Rejected'
        prob_approved = float(probabilities[0])
        prob_rejected = float(probabilities[1])

        # Save prediction to history
        save_prediction(session['user'], data, result, {
            'Approved': prob_approved,
            'Rejected': prob_rejected
        })
        
        return redirect(url_for('result', status=result, prob_approved=prob_approved, prob_rejected=prob_rejected))
    except Exception as e:
        flash('Error: ' + str(e))
        return redirect(url_for('home'))

@app.route('/result', methods=['GET'])
def result():
    if 'user' not in session:
        return redirect(url_for('login'))
    status = request.args.get('status')
    prob_approved = request.args.get('prob_approved')
    prob_rejected = request.args.get('prob_rejected')
    if not status or not prob_approved or not prob_rejected:
        flash('Invalid result data')
        return redirect(url_for('home'))
    return render_template('result.html', status=status, prob_approved=float(prob_approved), prob_rejected=float(prob_rejected))

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    history_data = load_history()
    user_history = history_data.get(session['user'], [])
    user_history.reverse()  # Show most recent first
    
    return render_template('history.html', history=user_history)

if __name__ == '__main__':
    app.run(debug=True)