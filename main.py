import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("acid_rain_pan_india_cleaned.csv")

df.drop(columns=["type"], inplace=True, errors="ignore")
df.drop_duplicates(inplace=True)

numeric_cols = df.select_dtypes(include=np.number).columns
non_numeric_cols = df.select_dtypes(exclude=np.number).columns

df[numeric_cols] = SimpleImputer(strategy="median").fit_transform(df[numeric_cols])
df[non_numeric_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[non_numeric_cols])

encoder = LabelEncoder()
for col in non_numeric_cols:
    df[col] = encoder.fit_transform(df[col])

plt.figure(figsize=(14,8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x="Season", y="pH", data=df)
plt.show()

top_so2 = df.groupby("Location")["SO2_Concentration_ppm"].mean().sort_values(ascending=False).head(10)
top_so2.plot(kind="bar")
plt.show()

X = df.drop(columns=["pH"])
y = df["pH"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print(r2_score(y_test, y_pred_lr))
print(mean_absolute_error(y_test, y_pred_lr))
print(mean_squared_error(y_test, y_pred_lr))

rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(r2_score(y_test, y_pred_rf))
print(mean_absolute_error(y_test, y_pred_rf))
print(mean_squared_error(y_test, y_pred_rf))

X_scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x="SO2_Concentration_ppm", y="pH", hue="Cluster", data=df, palette="viridis", s=10)
plt.show()
