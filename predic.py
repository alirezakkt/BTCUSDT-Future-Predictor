import numpy as np
import requests
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# fetch BTC Price
def fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=20):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract H, O, C, L prices & timestamps
        processed_data = np.array([
            [float(candle[2]), float(candle[1]), float(candle[4]), float(candle[3])] for candle in data
        ])
        target = np.array([float(candle[4]) for candle in data])
        timestamps = [datetime.fromtimestamp(int(candle[0]) / 1000) for candle in data]
        logging.info("Successfully fetched and processed api.")
        return processed_data, target, timestamps
    except Exception as e:
        logging.error(f"Error fetching API: {e}")
        raise

#Polynomial Regression Model with Incremental Updates
class PolynomialRegressionModel:
    def __init__(self, degree=2):
        self.degree = degree
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.model = SGDRegressor(max_iter=2000, tol=1e-4, alpha=0.001, eta0=0.01, learning_rate='adaptive')
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('poly_features', self.poly),
            ('sgd_regressor', self.model)
        ])

    def train(self, data, target):
        self.pipeline.fit(data, target)
        logging.info("Model training complete.")

    def predict(self, data):
        prediction = self.pipeline.predict(data)
        logging.info("Prediction made.")
        return prediction

    def update(self, new_data, new_target):
        poly_features = self.poly.transform(self.scaler.transform(new_data))
        self.model.partial_fit(poly_features, new_target)
        logging.info("Model updated with new data.")

# plot
def plot_predictions_with_future(actual, predicted, timestamps, future_predictions, future_timestamps, title="Market Price Predictions"):
    plt.figure(figsize=(14, 7))

    plt.plot(timestamps, actual, label="Actual Prices", marker='o', linestyle='-', color='blue', linewidth=1.5)
    plt.plot(timestamps, predicted, label="Predicted Prices", marker='x', linestyle='--', color='orange', linewidth=1.5)

    # Plot future
    for i, (future_time, future_price) in enumerate(zip(future_timestamps, future_predictions)):
        plt.scatter(future_time, future_price, color='green', s=100, edgecolors='black', label="Future Prediction" if i == 0 else "")
        plt.text(future_time, future_price + 0.01 * future_price, f"{future_price:.2f}", fontsize=9, ha='center', color='green')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("BTCUSDT Price", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def main():
    try:

        data, target, timestamps = fetch_binance_data()
        logging.info(f"Sample of fetched data: {data[:5]}")
        logging.info(f"Sample of target values: {target[:5]}")

        pr_model = PolynomialRegressionModel(degree=2)
        pr_model.train(data[:-1], target[:-1])

        predicted = pr_model.predict(data[:-1])

        X_future = np.array([data[-1]])
        future_prediction = pr_model.predict(X_future)[0]
        logging.info(f"Predicted Future Price: {future_prediction:.2f}")

        # Simulate timestamp
        future_timestamp = timestamps[-1] + timedelta(hours=1)

        plot_predictions_with_future(
            actual=target[:-1],
            predicted=predicted,
            timestamps=timestamps[:-1],
            future_predictions=[future_prediction],
            future_timestamps=[future_timestamp],
            title="Actual, Predicted, and Future Market Prices"
        )

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
