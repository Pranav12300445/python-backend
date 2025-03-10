import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS support
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def fetch_nav_data(ticker, period="1y"):
    """Fetch historical NAV data from Yahoo Finance with error handling"""
    try:
        fund = yf.Ticker(ticker)
        hist = fund.history(period=period)

        if hist.empty:
            return None  # Return None if no data is available

        df = hist[['Close']].reset_index()
        df.columns = ['Date', 'NAV']
        df['Date'] = df['Date'].astype(str)  # Convert Date to string for JSON response
        return df

    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None  # Return None if API request fails

def train_model(df):
    """Train a simple Linear Regression model to predict NAV"""
    if df is None or df.empty:
        return None  # Return None if no valid data

    df['Days'] = np.arange(len(df))  # Convert dates into numerical values
    X = df[['Days']]
    y = df['NAV']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

@app.route("/predict", methods=["GET"])
def predict_nav():
    """API endpoint to fetch NAV data and predict future NAV"""
    ticker = request.args.get("ticker", "VFIAX")  # Default to Vanguard 500 Index Fund

    df = fetch_nav_data(ticker)
    if df is None:
        return jsonify({"error": "Invalid mutual fund ticker or no data available"}), 400

    model = train_model(df)
    if model is None:
        return jsonify({"error": "Unable to train model due to insufficient data"}), 400

    # Predict NAV for next 7 days
    future_days = np.arange(len(df), len(df) + 7).reshape(-1, 1)
    predicted_nav = model.predict(future_days)

    # Format predictions
    future_dates = [(datetime.date.today() + datetime.timedelta(days=i)).isoformat() for i in range(1, 8)]
    predictions = [{"date": future_dates[i], "predicted_NAV": round(predicted_nav[i], 2)} for i in range(7)]

    return jsonify({
        "ticker": ticker,
        "historical_NAV": df.to_dict(orient="records"),
        "predictions": predictions
    })

if __name__ == "__main__":
    app.run(debug=True)
