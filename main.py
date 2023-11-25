import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

file_path = "C:/Users/45527/Downloads/beijing/2005-2015Beijing.xls"
weather_data = pd.read_excel(file_path, skiprows=6)
weather_data.head()

weather_data_processed = weather_data.copy()
weather_data_processed['T'] = pd.to_numeric(weather_data_processed['T'], errors='coerce')
weather_data_processed['RRR'] = pd.to_numeric(weather_data_processed['RRR'], errors='coerce')
weather_data_processed.dropna(subset=['T', 'RRR'], inplace=True)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(weather_data_processed['Date'], weather_data_processed['T'])
plt.title('Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature')

plt.subplot(1, 2, 2)
plt.plot(weather_data_processed['Date'], weather_data_processed['RRR'])
plt.title('Precipitation Over Time')
plt.xlabel('Date')
plt.ylabel('Precipitation')
plt.tight_layout()
plt.show()

model_data = weather_data_processed[['Year', 'Month', 'Day', 'Hour', 'T', 'RRR']]
scaler = StandardScaler()
model_data[['T', 'RRR']] = scaler.fit_transform(model_data[['T', 'RRR']])

X = model_data[['Year', 'Month', 'Day', 'Hour']]
y = model_data['T']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)
y_pred = linear_reg_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred)

random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)

cv_scores = cross_val_score(random_forest_model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_mse_scores = -cv_scores
cv_mse_scores_mean = np.mean(cv_mse_scores)
cv_mse_scores_std = np.std(cv_mse_scores)

feature_importances = random_forest_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
