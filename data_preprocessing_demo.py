

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class LoanDataPreprocessor:

    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_and_inspect_data(self, filepath):
       
        print("="*60)
        print("STEP 1: DATA LOADING AND INSPECTION")
        print("="*60)
        
        # Load dataset
        df = pd.read_csv(filepath)
        
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Basic information
        print(f"Dataset shape: {df.shape}")
        print(f"Total records: {df.shape[0]:,}")
        print(f"Total features: {df.shape[1]-1} (excluding target)")
        
        print("\nColumn Information:")
        print("-" * 40)
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col:<25} - {df[col].dtype}")
        
        print("\nFirst 5 records:")
        print(df.head())
        
        print("\nDataset Statistics:")
        print(df.describe())
        
        return df
    
    def analyze_data_quality(self, df):
      
        print("\n" + "="*60)
        print("STEP 2: DATA QUALITY ANALYSIS")
        print("="*60)
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print("Missing Values:")
        print("-" * 20)
        if missing_values.sum() == 0:
            print("✅ No missing values found!")
        else:
            print(missing_values[missing_values > 0])
        
        # Check data types
        print("\nData Types:")
        print("-" * 15)
        print(df.dtypes)
        
        # Target variable distribution
        print("\nTarget Variable Distribution:")
        print("-" * 35)
        target_dist = df['loan_status'].value_counts()
        print(target_dist)
        print(f"Approval Rate: {target_dist['Approved']/len(df)*100:.1f}%")
        print(f"Rejection Rate: {target_dist['Rejected']/len(df)*100:.1f}%")
        
        return df
    
    def handle_missing_values(self, df):
       
        print("\n" + "="*60)
        print("STEP 3: MISSING VALUE TREATMENT")
        print("="*60)
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            print("Applying forward fill method...")
            df = df.fillna(method='ffill')
            print("✅ Missing values handled")
        else:
            print("✅ No missing values to handle")
        
        return df
    
    def encode_categorical_variables(self, df):
        
        print("\n" + "="*60)
        print("STEP 4: CATEGORICAL VARIABLE ENCODING")
        print("="*60)
        
        # Define categorical columns
        categorical_cols = ['education', 'self_employed']
        
        print("Encoding mappings:")
        print("-" * 20)
        
        # Manual encoding for consistency
        education_map = {'Graduate': 0, 'Not Graduate': 1}
        self_employed_map = {'No': 0, 'Yes': 1}
        
        # Apply encodings
        df['education'] = df['education'].map(education_map)
        df['self_employed'] = df['self_employed'].map(self_employed_map)
        
        print("Education: Graduate → 0, Not Graduate → 1")
        print("Self Employed: No → 0, Yes → 1")
        
        # Verify encoding
        print("\nEncoding verification:")
        print(f"Education unique values: {sorted(df['education'].unique())}")
        print(f"Self Employed unique values: {sorted(df['self_employed'].unique())}")
        
        return df
    
    def analyze_feature_distributions(self, df):
        
        print("\n" + "="*60)
        print("STEP 5: FEATURE DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Numerical features analysis
        numerical_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 
                         'loan_term', 'cibil_score', 'residential_assets_value',
                         'commercial_assets_value', 'luxury_assets_value', 
                         'bank_asset_value']
        
        print("Numerical Features Statistics:")
        print("-" * 35)
        for col in numerical_cols:
            print(f"{col}:")
            print(f"  Range: {df[col].min():,} - {df[col].max():,}")
            print(f"  Mean: {df[col].mean():,.0f}")
            print(f"  Std: {df[col].std():,.0f}")
        
        # Categorical features analysis
        print("\nCategorical Features Distribution:")
        print("-" * 40)
        
        # Education distribution
        edu_dist = df['education'].value_counts()
        print("Education (encoded):")
        print(f"  Graduate (0): {edu_dist[0]} ({edu_dist[0]/len(df)*100:.1f}%)")
        print(f"  Not Graduate (1): {edu_dist[1]} ({edu_dist[1]/len(df)*100:.1f}%)")
        
        # Self employed distribution
        se_dist = df['self_employed'].value_counts()
        print("Self Employed (encoded):")
        print(f"  No (0): {se_dist[0]} ({se_dist[0]/len(df)*100:.1f}%)")
        print(f"  Yes (1): {se_dist[1]} ({se_dist[1]/len(df)*100:.1f}%)")
        
        return df
    
    def prepare_features_and_target(self, df):
       
        print("\n" + "="*60)
        print("STEP 6: FEATURE AND TARGET PREPARATION")
        print("="*60)
        
        # Define features (exclude loan_id and loan_status)
        feature_columns = [col for col in df.columns if col not in ['loan_id', 'loan_status']]
        self.feature_names = feature_columns
        
        # Prepare features and target
        X = df[feature_columns]
        y = df['loan_status'].map({'Approved': 0, 'Rejected': 1})
        
        print(f"Features selected: {len(feature_columns)}")
        print("Feature list:")
        for i, feature in enumerate(feature_columns, 1):
            print(f"  {i:2d}. {feature}")
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        print("\nTarget encoding: Approved → 0, Rejected → 1")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
       
        print("\n" + "="*60)
        print("STEP 7: DATA SPLITTING")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]:,} samples ({(1-test_size)*100:.0f}%)")
        print(f"Testing set: {X_test.shape[0]:,} samples ({test_size*100:.0f}%)")
        print(f"Random state: {random_state}")
        
        # Check class distribution in splits
        print("\nClass distribution in splits:")
        print("Training set:", y_train.value_counts().to_dict())
        print("Testing set:", y_test.value_counts().to_dict())
        
        return X_train, X_test, y_train, y_test
    
    def validate_preprocessing(self, X_train, X_test, y_train, y_test):
       
        print("\n" + "="*60)
        print("STEP 8: PREPROCESSING VALIDATION")
        print("="*60)
        
        # Check for any remaining issues
        checks = []
        
        # Check for missing values
        train_missing = X_train.isnull().sum().sum()
        test_missing = X_test.isnull().sum().sum()
        checks.append(("No missing values in training set", train_missing == 0))
        checks.append(("No missing values in testing set", test_missing == 0))
        
        # Check data types
        numeric_types = X_train.dtypes.apply(lambda x: x in ['int64', 'float64']).all()
        checks.append(("All features are numeric", numeric_types))
        
        # Check feature consistency
        feature_consistency = list(X_train.columns) == list(X_test.columns)
        checks.append(("Feature consistency between train/test", feature_consistency))
        
        # Check target encoding
        target_values = set(y_train.unique()) | set(y_test.unique())
        target_binary = target_values == {0, 1}
        checks.append(("Target properly encoded (0,1)", target_binary))
        
        print("Validation Results:")
        print("-" * 20)
        for check_name, result in checks:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {check_name}")
        
        return all(result for _, result in checks)
    
    def create_sample_prediction_pipeline(self, X_train, y_train):
       
        print("\n" + "="*60)
        print("STEP 9: SAMPLE PREDICTION PIPELINE")
        print("="*60)
        
        # Train a simple model for demonstration
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        print("✅ Sample Random Forest model trained")
        
        # Show feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Most Important Features:")
        print("-" * 35)
        for i, (_, row) in enumerate(feature_importance.head().iterrows(), 1):
            print(f"{i}. {row['feature']:<25} {row['importance']:.4f}")
        
        return model
    
    def demonstrate_real_time_preprocessing(self):
        
        print("\n" + "="*60)
        print("STEP 10: REAL-TIME PREPROCESSING DEMO")
        print("="*60)
        
        # Sample input data (as would come from web form)
        sample_input = {
            'no_of_dependents': 2,
            'education': 'Graduate',
            'self_employed': 'No',
            'income_annum': 5000000,
            'loan_amount': 15000000,
            'loan_term': 12,
            'cibil_score': 750,
            'residential_assets_value': 8000000,
            'commercial_assets_value': 3000000,
            'luxury_assets_value': 12000000,
            'bank_asset_value': 4000000
        }
        
        print("Sample input from web form:")
        for key, value in sample_input.items():
            print(f"  {key}: {value}")
        
        # Preprocessing function for real-time use
        def preprocess_input(user_data):
            education_map = {'Graduate': 0, 'Not Graduate': 1}
            self_employed_map = {'No': 0, 'Yes': 1}
            
            features = [
                int(user_data['no_of_dependents']),
                education_map[user_data['education']],
                self_employed_map[user_data['self_employed']],
                float(user_data['income_annum']),
                float(user_data['loan_amount']),
                int(user_data['loan_term']),
                int(user_data['cibil_score']),
                float(user_data['residential_assets_value']),
                float(user_data['commercial_assets_value']),
                float(user_data['luxury_assets_value']),
                float(user_data['bank_asset_value'])
            ]
            return features
        
        # Apply preprocessing
        processed_features = preprocess_input(sample_input)
        
        print("\nProcessed feature vector:")
        print(processed_features)
        print(f"Vector length: {len(processed_features)} features")
        
        return processed_features

def main():
    
    print("LOAN APPROVAL PREDICTION - DATA PREPROCESSING PIPELINE")
    print("=" * 70)
   
    preprocessor = LoanDataPreprocessor()
    
    try:
       
        df = preprocessor.load_and_inspect_data('loan_approval_dataset.csv')
        df = preprocessor.analyze_data_quality(df)
        df = preprocessor.handle_missing_values(df)
        df = preprocessor.encode_categorical_variables(df)
        df = preprocessor.analyze_feature_distributions(df)
        X, y = preprocessor.prepare_features_and_target(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        validation_passed = preprocessor.validate_preprocessing(X_train, X_test, y_train, y_test)
        
        if validation_passed:
            model = preprocessor.create_sample_prediction_pipeline(X_train, y_train)
            preprocessor.demonstrate_real_time_preprocessing()
            
            print("\n" + "="*70)
            print("✅ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print("Data is now ready for machine learning model training.")
            print(f"Training samples: {X_train.shape[0]:,}")
            print(f"Testing samples: {X_test.shape[0]:,}")
            print(f"Features: {X_train.shape[1]}")
            print("All preprocessing steps validated and working correctly.")
            
        else:
            print("\n❌ PREPROCESSING VALIDATION FAILED!")
            print("Please check the preprocessing steps.")
            
    except FileNotFoundError:
        print("❌ Error: loan_approval_dataset.csv not found!")
        print("Please ensure the dataset file is in the same directory.")
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")

if __name__ == "__main__":
    main()