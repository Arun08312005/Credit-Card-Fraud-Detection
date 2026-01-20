import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        
    def load_dataset(self, sample_size=10000):
        """Generate synthetic fraud detection data"""
        print("Generating synthetic fraud detection dataset...")
        
        # Generate synthetic data
        np.random.seed(42)
        n_samples = sample_size
        
        # Features (28 features like credit card dataset)
        features = []
        for i in range(28):
            if i < 14:  # V1-V14: Normally distributed
                features.append(np.random.randn(n_samples))
            else:  # V15-V28: Uniform distributed
                features.append(np.random.uniform(-5, 5, n_samples))
        
        # Amount and Time
        amount = np.random.exponential(100, n_samples)
        time = np.random.uniform(0, 172800, n_samples)  # 48 hours in seconds
        
        # Create fraud patterns
        fraud_mask = np.zeros(n_samples, dtype=bool)
        
        # Rule 1: Very high amounts (> 1000) have 30% fraud chance
        high_amount_mask = amount > 1000
        fraud_mask[high_amount_mask] = np.random.rand(np.sum(high_amount_mask)) < 0.3
        
        # Rule 2: Late night transactions (12AM-6AM) have 20% fraud chance
        hour = (time % 86400) / 3600
        night_mask = (hour >= 0) & (hour <= 6)
        fraud_mask[night_mask] = np.random.rand(np.sum(night_mask)) < 0.2
        
        # Rule 3: Create some feature patterns for fraud
        for i in [1, 5, 10, 14, 20]:
            extreme_mask = np.abs(features[i]) > 3
            fraud_mask[extreme_mask] = np.random.rand(np.sum(extreme_mask)) < 0.4
        
        # Add some noise
        fraud_mask = fraud_mask | (np.random.rand(n_samples) < 0.01)
        
        # Create labels (1 for fraud, 0 for legitimate)
        labels = fraud_mask.astype(int)
        
        # Create DataFrame
        columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
        data = np.column_stack([time] + features + [amount, labels])
        df = pd.DataFrame(data, columns=columns)
        
        print(f"Generated dataset: {df.shape}")
        print(f"Fraud rate: {df['Class'].mean():.2%}")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        print("Preprocessing data...")
        
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Scale Amount column
        if 'Amount' in X.columns:
            from sklearn.preprocessing import RobustScaler
            scaler_amount = RobustScaler()
            X['Amount'] = scaler_amount.fit_transform(X['Amount'].values.reshape(-1, 1))
        
        # Scale Time column
        if 'Time' in X.columns:
            X['Time'] = (X['Time'] - X['Time'].mean()) / X['Time'].std()
        
        # Scale all features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_model(self, sample_size=10000):
        """Train the fraud detection model"""
        print("="*60)
        print("Training Fraud Detection Model")
        print("="*60)
        
        # Generate and load dataset
        df = self.load_dataset(sample_size)
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Fraud rate in train: {y_train.mean():.2%}")
        print(f"Fraud rate in test: {y_test.mean():.2%}")
        
        # Create and train model
        print("\nTraining Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"\nModel Performance:")
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Testing Accuracy: {test_score:.4f}")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import classification_report, roc_auc_score
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        
        # Save model
        self.save_model()
        
        self.is_trained = True
        
        return {
            'accuracy': test_score,
            'roc_auc': roc_auc,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'fraud_rate': y.mean()
        }
    
    def save_model(self):
        """Save trained model and scaler"""
        print("\nSaving model...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save model
        joblib.dump(self.model, 'models/fraud_model.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        print("✓ Model saved successfully!")
        print("  - models/fraud_model.pkl")
        print("  - models/scaler.pkl")
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load('models/fraud_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.is_trained = True
            print("✓ Model loaded successfully!")
            return True
        except FileNotFoundError:
            print("✗ Model not found. Please train the model first.")
            return False
    
    def predict(self, features):
        """Predict fraud probability"""
        if not self.is_trained:
            if not self.load_model():
                return self._get_random_prediction()
        
        try:
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0][1]
            
            return {
                'is_fraud': bool(prediction),
                'probability': float(probability),
                'risk_score': int(probability * 100),
                'confidence': 'high' if probability > 0.8 else 'medium' if probability > 0.6 else 'low'
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._get_random_prediction()
    
    def _get_random_prediction(self):
        """Get random prediction for demo"""
        prob = np.random.random()
        is_fraud = prob > 0.7
        
        return {
            'is_fraud': is_fraud,
            'probability': float(prob),
            'risk_score': int(prob * 100),
            'confidence': 'demo'
        }
    
    def generate_sample_transaction(self):
        """Generate a sample transaction for testing"""
        # Generate random features
        features = []
        
        # Time (0-172800 seconds)
        features.append(np.random.uniform(0, 172800))
        
        # V1-V28 features
        for i in range(28):
            if i < 14:
                features.append(np.random.randn())
            else:
                features.append(np.random.uniform(-5, 5))
        
        # Amount (exponential distribution)
        features.append(np.random.exponential(100))
        
        return features

if __name__ == "__main__":
    # Train model for testing
    model = FraudDetectionModel()
    results = model.train_model(sample_size=5000)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    
    # Test with sample transaction
    sample_features = model.generate_sample_transaction()
    prediction = model.predict(sample_features)
    
    print("\nSample Prediction:")
    print(f"Fraud: {prediction['is_fraud']}")
    print(f"Probability: {prediction['probability']:.4f}")
    print(f"Risk Score: {prediction['risk_score']}/100")