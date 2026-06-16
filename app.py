
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

app = Flask(__name__)

# =====================================
# Load Dataset & Model
# =====================================

df = pd.read_csv("data/AmesHousing.csv")
model = joblib.load("model.pkl")

try:
    actual_pred = pd.read_csv(
        "data/actual_vs_predicted.csv"
    )
except:
    actual_pred = pd.DataFrame({
        "Actual": [],
        "Predicted": []
    })

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
    X,
    y,
    test_size=0.2,
    random_state=42
)

predictions = model.predict(X_test)

r2_value = round(
    r2_score(y_test, predictions),
    3
)

rmse_value = round(
    np.sqrt(
        mean_squared_error(y_test, predictions)
    ),
    2
)

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=True
)

# =====================================
# Home Page
# =====================================

@app.route("/")
def home():
    return render_template("index.html")


# =====================================
# Prediction
# =====================================

@app.route("/predict", methods=["POST"])
def predict():

    features_input = np.array([[

        float(request.form["overall_qual"]),
        float(request.form["gr_liv_area"]),
        float(request.form["garage_cars"]),
        float(request.form["garage_area"]),
        float(request.form["full_bath"]),
        float(request.form["year_built"])

    ]])

    prediction = model.predict(features_input)

    return render_template(
        "index.html",
        prediction_text=f"Estimated House Price: ${prediction[0]:,.2f}"
    )


# =====================================
# Dashboard
# =====================================

@app.route("/dashboard")
def dashboard():

    avg_price = round(df["SalePrice"].mean())
    max_price = round(df["SalePrice"].max())
    total_houses = len(df)
    avg_quality = round(df["Overall Qual"].mean(), 1)

    # =====================================
    # Histogram
    # =====================================

    fig1 = px.histogram(
        df,
        x="SalePrice",
        nbins=30,
        title="House Price Distribution",
        template="plotly_dark"
    )

    fig1.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#111827",
        font=dict(color="white"),
        height=450,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )

    # =====================================
    # Scatter Plot
    # =====================================

    fig2 = px.scatter(
        df,
        x="Gr Liv Area",
        y="SalePrice",
        color="Overall Qual",
        title="Living Area vs Sale Price",
        template="plotly_dark"
    )

    fig2.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#111827",
        font=dict(color="white"),
        height=450,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )

    # =====================================
    # Heatmap
    # =====================================

    corr = df[
        [
            "Overall Qual",
            "Gr Liv Area",
            "Garage Cars",
            "Garage Area",
            "Full Bath",
            "Year Built",
            "SalePrice"
        ]
    ].corr()

    fig3 = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Turbo",
        title="Feature Correlation Heatmap",
        template="plotly_dark"
    )

    fig3.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#111827",
        font=dict(color="white"),
        height=650
    )

    # =====================================
    # Actual vs Predicted
    # =====================================

    if len(actual_pred) > 0:

        fig4 = px.scatter(
            actual_pred,
            x="Actual",
            y="Predicted",
            title="Actual vs Predicted Prices",
            template="plotly_dark"
        )

        fig4.add_shape(
            type="line",
            x0=actual_pred["Actual"].min(),
            y0=actual_pred["Actual"].min(),
            x1=actual_pred["Actual"].max(),
            y1=actual_pred["Actual"].max(),
            line=dict(
                color="red",
                width=3
            )
        )

    else:

        fig4 = px.scatter(
            title="Actual vs Predicted Prices"
        )

    fig4.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#111827",
        font=dict(color="white"),
        height=500,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )

    # =====================================
    # Feature Importance
    # =====================================

    fig5 = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        title="Feature Importance",
        template="plotly_dark"
    )

    fig5.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#111827",
        font=dict(color="white"),
        height=500
    )

    return render_template(

        "dashboard.html",

        avg_price=avg_price,
        max_price=max_price,
        total_houses=total_houses,
        avg_quality=avg_quality,

        r2_score_value=r2_value,
        rmse_score_value=rmse_value,

        price_chart=fig1.to_html(
            full_html=False,
            config={
                "responsive": True,
                "displaylogo": False
            }
        ),

        scatter_chart=fig2.to_html(
            full_html=False,
            config={
                "responsive": True,
                "displaylogo": False
            }
        ),

        heatmap=fig3.to_html(
            full_html=False,
            config={
                "responsive": True,
                "displaylogo": False
            }
        ),

        actual_vs_predicted=fig4.to_html(
            full_html=False,
            config={
                "responsive": True,
                "displaylogo": False
            }
        ),

        feature_importance_chart=fig5.to_html(
            full_html=False,
            config={
                "responsive": True,
                "displaylogo": False
            }
        )

    )


if __name__ == "__main__":
    app.run(debug=True)

