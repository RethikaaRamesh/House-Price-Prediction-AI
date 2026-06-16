import pandas as pd
import plotly.express as px

df = pd.read_csv("data/AmesHousing.csv")

# Price Distribution

fig1 = px.histogram(
    df,
    x="SalePrice",
    nbins=30,
    title="House Price Distribution"
)

fig1.write_html(
    "templates/price_distribution.html"
)

# Living Area vs Price

fig2 = px.scatter(
    df,
    x="Gr Liv Area",
    y="SalePrice",
    color="Overall Qual",
    title="Living Area vs Sale Price"
)

fig2.write_html(
    "templates/scatter_plot.html"
)

print("Charts Created")