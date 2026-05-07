import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Global cache
trained_models = {}

ALPHA_VANTAGE_API_KEY = 'K7WQV2SFCRFE0AKC'
LOOKBACK = 20
EPOCHS = 10

def fetch_stock_data(symbol, period="6mo"):
    """Fetch stock data from Yahoo Finance (more reliable than Alpha Vantage)."""
    print(f"  📡 Fetching data for {symbol}...")
    
    # Use yfinance directly (more reliable)
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    
    if df.empty:
        raise ValueError(f"No data available for {symbol}")
    
    # Format to match expected structure
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    print(f"  ✅ Retrieved {len(df)} days of data")
    return df, "Yahoo Finance"

def prepare_lstm_data(df, lookback=LOOKBACK):
    """Prepare sequences for LSTM."""
    prices = df["close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)
    
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y, scaler

def build_lstm_model(lookback=LOOKBACK):
    """Build LSTM model."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def calculate_accuracy(y_true, y_pred):
    """Calculate directional accuracy."""
    if len(y_true) < 2:
        return 50.0
    
    true_direction = np.sign(np.diff(y_true.flatten()))
    pred_direction = np.sign(np.diff(y_pred.flatten()))
    
    if len(true_direction) == 0:
        return 50.0
    
    accuracy = np.mean(true_direction == pred_direction) * 100
    return accuracy

def train_lstm(symbol):
    """Train LSTM model."""
    symbol = symbol.upper()
    
    if symbol in trained_models:
        print(f"  🔄 Using cached model for {symbol}")
        return trained_models[symbol]
    
    print(f"\n{'='*50}")
    print(f"  🚀 Training LSTM model for {symbol}")
    print(f"{'='*50}")
    
    # Fetch data
    df, data_source = fetch_stock_data(symbol)
    
    if len(df) < LOOKBACK + 10:
        raise ValueError(f"Insufficient data for {symbol}")
    
    # Prepare data
    X, y, scaler = prepare_lstm_data(df)
    
    # Split data
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build and train model
    model = build_lstm_model()
    
    print(f"  🧠 Training on {len(X_train)} sequences...")
    
    # Train with progress display
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1  # Show progress
    )
    
    # Evaluate
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_real = scaler.inverse_transform(y_pred_scaled)
    
    accuracy = calculate_accuracy(y_test_real, y_pred_real)
    
    print(f"\n✅ Model accuracy: {accuracy:.1f}%")
    print(f"{'='*50}\n")
    
    result = (model, scaler, df, accuracy, data_source)
    trained_models[symbol] = result
    return result

def predict_price(symbol):
    """Predict next day's price."""
    symbol = symbol.upper()
    model, scaler, df, accuracy, data_source = train_lstm(symbol)
    
    # Prepare last sequence
    last_sequence = df["close"].values[-LOOKBACK:].reshape(-1, 1)
    last_scaled = scaler.transform(last_sequence)
    last_scaled = last_scaled.reshape(1, LOOKBACK, 1)
    
    # Predict
    pred_scaled = model.predict(last_scaled, verbose=0)
    predicted_price = float(scaler.inverse_transform(pred_scaled)[0][0])
    current_price = float(df["close"].iloc[-1])
    
    change_percent = ((predicted_price - current_price) / current_price) * 100
    
    if change_percent > 2:
        suggestion = "BUY"
    elif change_percent < -2:
        suggestion = "SELL"
    else:
        suggestion = "HOLD"
    
    # Historical data for chart
    hist = df["close"].tail(90)
    historical_prices = [round(float(p), 2) for p in hist.values]
    dates = [str(d.date()) for d in hist.index]
    
    return {
        "symbol": symbol,
        "predicted_price": round(predicted_price, 2),
        "current_price": round(current_price, 2),
        "change_percent": round(change_percent, 2),
        "suggestion": suggestion,
        "accuracy_score": round(accuracy, 1),
        "data_source_used": data_source,
        "historical_prices": historical_prices,
        "dates": dates,
    }