import joblib
model = joblib.load("model/optimized_heart_model.pkl")   # or optimized_heart_model.pkl
print("model object:", model)
print("type:", type(model))
print("has named_steps?:", hasattr(model, "named_steps"))
print("has predict_proba?:", hasattr(model, "predict_proba"))