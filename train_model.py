import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv("data/heart.csv")

# Define features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define preprocessing
categorical_features = ['cp', 'restecg', 'slope', 'thal']
numeric_features = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Create pipeline with Gradient Boosting (better for imbalanced data)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=5
    ))
])

# Fit the model
pipeline.fit(X_train, y_train)

# Calibrate probabilities for better risk estimation
calibrated_clf = CalibratedClassifierCV(
    pipeline,
    method='sigmoid',
    cv='prefit'
)
calibrated_clf.fit(X_train, y_train)

# Evaluate
y_pred = calibrated_clf.predict(X_test)
y_proba = calibrated_clf.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"\nROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# Save model and preprocessor
joblib.dump(calibrated_clf, "model/optimized_heart_model.pkl")
joblib.dump(preprocessor, "model/optimized_preprocessor.pkl")

# Save feature names for inference model
feature_names = (numeric_features +
                 list(pipeline.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .get_feature_names_out(categorical_features)))
joblib.dump(feature_names, "model/optimized_feature_names.pkl")

print("\nModel saved successfully!")