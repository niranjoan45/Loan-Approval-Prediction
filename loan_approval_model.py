import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('loan_approval_dataset.csv')

# Strip column names to remove leading/trailing spaces
df.columns = df.columns.str.strip()

# Display basic info
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nLoan status distribution:")
print(df['loan_status'].value_counts())

# Fill missing values if any (simple approach)
df.fillna(method='ffill', inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['education', 'self_employed', 'loan_status']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop(['loan_id', 'loan_status'], axis=1)
y = df['loan_status']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

# Train Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Save the best model (assuming Random Forest performs better)
joblib.dump(rf, 'loan_approval_rf_model.pkl')
print("\nRandom Forest model saved as loan_approval_rf_model.pkl")
