import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split

df = pd.read_csv("data/AmesHousing.csv")

features = [
    'Overall Qual',
    'Gr Liv Area',
    'Garage Cars',
    'Garage Area',
    'Full Bath',
    'Year Built'
]

# Price Distribution
plt.figure(figsize=(8,5))
plt.hist(df['SalePrice'], bins=30)
plt.title("Price Distribution")
plt.savefig("static/price_distribution.png")
plt.close()

# Feature Importance
model = joblib.load("model.pkl")

importance = model.feature_importances_

plt.figure(figsize=(8,5))
plt.bar(features, importance)
plt.xticks(rotation=20)
plt.title("Feature Importance")
plt.savefig("static/feature_importance.png")
plt.close()

# Actual vs Predicted
X = df[features]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

pred = model.predict(X_test)

plt.figure(figsize=(8,5))
plt.scatter(y_test,pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.savefig("static/actual_vs_predicted.png")
plt.close()