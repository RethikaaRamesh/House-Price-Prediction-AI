from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import joblib
import numpy as np

df = pd.read_csv("data/AmesHousing.csv")

features = [
    "Overall Qual",
    "Gr Liv Area",
    "Garage Cars",
    "Garage Area",
    "Full Bath",
    "Year Built"
]

X = df[features]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("R2 Score:", r2)
print("RMSE:", rmse)

joblib.dump(model, "model.pkl")

actual_pred = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

actual_pred.to_csv(
    "data/actual_vs_predicted.csv",
    index=False
)