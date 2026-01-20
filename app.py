from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
import numpy as np
import threading
import time
from model_training import FraudDetectionModel

app = Flask(__name__, 
            static_folder='../frontend',
            template_folder='../frontend',
            static_url_path='')

CORS(app)

# Initialize model
fraud_model = FraudDetectionModel()

# Track training status
training_in_progress = False
training_results = None

# Routes for serving frontend files
@app.route('/')
def serve_index():
    return render_template('index.html')

@app.route('/dashboard')
def serve_dashboard():
    return render_template('dashboard.html')

@app.route('/css/<path:path>')
def serve_css(path):
    return send_from_directory('../frontend/css', path)

@app.route('/js/<path:path>')
def serve_js(path):
    return send_from_directory('../frontend/js', path)

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Fraud Detection API',
        'version': '1.0.0',
        'model_loaded': fraud_model.is_trained,
        'timestamp': time.time()
    })

@app.route('/api/model/status', methods=['GET'])
def model_status():
    """Get model status"""
    return jsonify({
        'is_trained': fraud_model.is_trained,
        'training_in_progress': training_in_progress,
        'has_model_files': os.path.exists('models/fraud_model.pkl'),
        'timestamp': time.time()
    })

@app.route('/api/model/train', methods=['POST'])
def train_model():
    """Start model training"""
    global training_in_progress, training_results
    
    if training_in_progress:
        return jsonify({
            'success': False,
            'error': 'Training already in progress'
        }), 400
    
    # Start training in background
    def train_background():
        global training_in_progress, training_results
        training_in_progress = True
        
        try:
            results = fraud_model.train_model(sample_size=5000)
            training_results = results
        except Exception as e:
            print(f"Training error: {e}")
        finally:
            training_in_progress = False
    
    thread = threading.Thread(target=train_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Training started in background',
        'training_id': int(time.time())
    })

@app.route('/api/model/load', methods=['POST'])
def load_model():
    """Load pre-trained model"""
    try:
        success = fraud_model.load_model()
        return jsonify({
            'success': success,
            'message': 'Model loaded successfully' if success else 'Model not found'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict fraud for transaction"""
    try:
        data = request.json
        
        if 'features' in data:
            # Use provided features
            features = data['features']
        else:
            # Generate random features
            features = fraud_model.generate_sample_transaction()
            
            # Override with provided values if available
            if 'amount' in data:
                features[-1] = float(data['amount'])
        
        # Get prediction
        prediction = fraud_model.predict(features)
        
        # Add transaction details
        hour = int((features[0] % 86400) / 3600)
        time_of_day = "Night" if hour < 6 else "Morning" if hour < 12 else "Afternoon" if hour < 18 else "Evening"
        
        prediction['transaction_details'] = {
            'id': f"TX{np.random.randint(10000, 99999)}",
            'amount': f"${features[-1]:.2f}",
            'time': f"{hour:02d}:{np.random.randint(0, 60):02d}",
            'time_of_day': time_of_day,
            'location': np.random.choice(['New York', 'London', 'Tokyo', 'Sydney', 'Paris']),
            'merchant': np.random.choice(['Amazon', 'Walmart', 'Uber', 'Netflix', 'Apple'])
        }
        
        # Add recommendation
        risk_score = prediction['risk_score']
        if risk_score >= 80:
            recommendation = {
                'action': 'BLOCK',
                'message': 'High fraud risk. Block transaction and alert security.',
                'color': '#FF6B6B'
            }
        elif risk_score >= 60:
            recommendation = {
                'action': 'REVIEW',
                'message': 'Suspicious transaction. Requires manual review.',
                'color': '#FFD166'
            }
        elif risk_score >= 40:
            recommendation = {
                'action': 'MONITOR',
                'message': 'Moderate risk. Process with additional verification.',
                'color': '#4ECDC4'
            }
        else:
            recommendation = {
                'action': 'APPROVE',
                'message': 'Low risk. Transaction appears legitimate.',
                'color': '#06D6A0'
            }
        
        prediction['recommendation'] = recommendation
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/demo/predict', methods=['GET'])
def demo_predict():
    """Demo prediction endpoint"""
    try:
        # Generate random transaction
        features = fraud_model.generate_sample_transaction()
        
        # Make it more likely to be fraud for demo
        if np.random.random() < 0.3:
            features[-1] = np.random.exponential(5000)  # High amount
            features[1] = np.random.randn() * 5  # Extreme V1
        
        prediction = fraud_model.predict(features)
        
        # Add demo details
        hour = int((features[0] % 86400) / 3600)
        time_of_day = "Night" if hour < 6 else "Morning" if hour < 12 else "Afternoon" if hour < 18 else "Evening"
        
        merchants = {
            'low': ['Amazon', 'Walmart', 'Uber', 'Netflix', 'Starbucks'],
            'high': ['Luxury Store', 'Unknown Merchant', 'Intl. Transfer', 'Crypto Exchange']
        }
        
        amount = features[-1]
        merchant_type = 'high' if amount > 1000 or prediction['risk_score'] > 70 else 'low'
        
        prediction['transaction'] = {
            'id': f"DEMO{np.random.randint(1000, 9999)}",
            'amount': f"${amount:.2f}",
            'merchant': np.random.choice(merchants[merchant_type]),
            'location': np.random.choice(['New York', 'London', 'Tokyo', 'Moscow', 'Lagos']),
            'time': f"{hour:02d}:{np.random.randint(0, 60):02d}",
            'time_of_day': time_of_day
        }
        
        # Add recommendation
        risk_score = prediction['risk_score']
        if risk_score >= 80:
            recommendation = {
                'action': 'BLOCK',
                'message': 'High fraud risk detected!',
                'color': '#FF6B6B'
            }
        elif risk_score >= 60:
            recommendation = {
                'action': 'REVIEW',
                'message': 'Suspicious transaction detected.',
                'color': '#FFD166'
            }
        else:
            recommendation = {
                'action': 'APPROVE',
                'message': 'Transaction appears legitimate.',
                'color': '#06D6A0'
            }
        
        prediction['recommendation'] = recommendation
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/transactions/recent', methods=['GET'])
def get_recent_transactions():
    """Get recent transaction history"""
    transactions = []
    
    for i in range(10):
        features = fraud_model.generate_sample_transaction()
        prediction = fraud_model.predict(features)
        
        hour = int((features[0] % 86400) / 3600)
        
        transactions.append({
            'id': f"TX{10000 + i}",
            'amount': f"${features[-1]:.2f}",
            'merchant': np.random.choice(['Amazon', 'Walmart', 'Uber', 'Netflix', 'Apple']),
            'time': f"{hour:02d}:{np.random.randint(0, 60):02d}",
            'status': 'fraud' if prediction['is_fraud'] else 'legitimate',
            'risk_score': prediction['risk_score']
        })
    
    return jsonify({
        'success': True,
        'transactions': transactions
    })

if __name__ == '__main__':
    print("="*60)
    print("Fraud Detection System")
    print("="*60)
    
    # Try to load existing model
    if os.path.exists('models/fraud_model.pkl'):
        print("Loading existing model...")
        fraud_model.load_model()
    else:
        print("No model found. Train using the web interface.")
    
    print(f"\nModel status: {'Loaded' if fraud_model.is_trained else 'Not loaded'}")
    print(f"Server starting on http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, port=5000, host='0.0.0.0')