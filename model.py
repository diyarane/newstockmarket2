import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Global cache
trained_models = {}

LOOKBACK = 50
EPOCHS = 100

def fetch_stock_data(symbol, period="2y"):
    """Fetch stock data with maximum features."""
    print(f"  📡 Fetching data for {symbol}...")
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    
    if df.empty:
        raise ValueError(f"No data available for {symbol}")
    
    # Create comprehensive features
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log1p(df['returns'])
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    df['open_close_ratio'] = (df['open'] - df['close']) / df['close']
    
    # Moving averages and crossovers
    for period_ma in [5, 10, 20, 30, 50]:
        df[f'ma_{period_ma}'] = df['close'].rolling(window=period_ma).mean()
        df[f'ma_ratio_{period_ma}'] = df['close'] / df[f'ma_{period_ma}'] - 1
    
    # Exponential moving averages
    for span in [12, 26]:
        df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume features
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    df['volume_price_trend'] = df['volume'] * df['returns']
    df['obv'] = (np.sign(df['returns']) * df['volume']).cumsum()
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['atr'] = df['high'] - df['low']
    df['atr'] = df['atr'].rolling(window=14).mean()
    
    # Price momentum
    for period_mom in [5, 10, 20]:
        df[f'momentum_{period_mom}'] = df['close'].pct_change(period_mom)
    
    # Lagged features (previous day's values)
    for col in ['close', 'returns', 'volume', 'volatility']:
        for lag in [1, 2, 3, 5]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Drop NaN values
    df = df.dropna()
    
    print(f"  ✅ Retrieved {len(df)} days with {len(df.columns)} technical indicators")
    return df

def prepare_lstm_data(df, lookback=LOOKBACK):
    """Prepare sequences with multiple features."""
    # Select best features for prediction
    feature_cols = [
        'close', 'returns', 'log_returns', 'high_low_ratio', 'open_close_ratio',
        'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_30',
        'macd', 'macd_signal', 'macd_histogram', 'rsi', 'bb_position',
        'volume_ratio', 'volatility', 'momentum_5', 'momentum_10',
        'close_lag_1', 'close_lag_2', 'returns_lag_1'
    ]
    
    # Filter available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"  📊 Using {len(available_cols)} features for prediction")
    
    # Scale features
    scaler_X = RobustScaler()
    scaled_features = scaler_X.fit_transform(df[available_cols])
    
    # Target is next day's return
    target = df['returns'].shift(-1).values[:-1]
    scaled_features = scaled_features[:-1]
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_features)):
        X.append(scaled_features[i - lookback:i, :])
        y.append(target[i])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler_X, len(available_cols)

def build_lstm_model(input_shape, n_features):
    """Build advanced LSTM model optimized for M4 Mac."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
    from tensorflow.keras.optimizers import legacy  # Use legacy optimizer for M4
    from tensorflow.keras.regularizers import l2
    
    model = Sequential([
        # First bidirectional LSTM layer
        Bidirectional(LSTM(128, return_sequences=True, 
                          input_shape=input_shape,
                          kernel_regularizer=l2(0.0001))),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second bidirectional LSTM layer
        Bidirectional(LSTM(64, return_sequences=True,
                          kernel_regularizer=l2(0.0001))),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third LSTM layer
        LSTM(32, return_sequences=False,
             kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Dense layers
        Dense(64, activation='relu', kernel_regularizer=l2(0.0001)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1, activation='tanh')
    ])
    
    optimizer = legacy.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
    
    return model

def calculate_directional_accuracy(y_true, y_pred):
    """Calculate directional accuracy."""
    if len(y_true) < 2:
        return 50.0
    
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    
    correct = np.sum(true_direction == pred_direction)
    accuracy = (correct / len(y_true)) * 100
    
    return accuracy

def train_lstm(symbol):
    """Train advanced LSTM model optimized for M4 Mac."""
    symbol = symbol.upper()
    
    if symbol in trained_models:
        print(f"  🔄 Using cached model for {symbol}")
        return trained_models[symbol]
    
    print(f"\n{'='*50}")
    print(f"  🚀 Training Advanced LSTM Model for {symbol}")
    print(f"{'='*50}")
    
    # Fetch data with features
    df = fetch_stock_data(symbol)
    
    if len(df) < LOOKBACK + 100:
        raise ValueError(f"Insufficient data for {symbol}. Need at least {LOOKBACK + 100} days.")
    
    # Prepare data
    X, y, scaler_X, n_features = prepare_lstm_data(df)
    
    if len(X) == 0:
        raise ValueError("Not enough data after preprocessing")
    
    # Train final model on all data
    print(f"  🧠 Training final model on {len(X)} sequences...")
    final_model = build_lstm_model((LOOKBACK, n_features), n_features)
    
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    
    history = final_model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stop, reduce_lr],
        verbose=1  # Show progress so you can see it's working
    )
    
    # Calculate final accuracy
    y_pred_final = final_model.predict(X, verbose=0).flatten()
    final_accuracy = calculate_directional_accuracy(y, y_pred_final)
    
    print(f"\n✅ Model Performance for {symbol}:")
    print(f"  • Directional Accuracy: {final_accuracy:.1f}%")
    print(f"  • Best Validation Loss: {min(history.history['val_loss']):.4f}")
    print(f"{'='*50}\n")
    
    result = (final_model, scaler_X, df, final_accuracy, n_features)
    trained_models[symbol] = result
    return result

def predict_price(symbol):
    """Predict next day's price movement."""
    symbol = symbol.upper()
    
    try:
        model, scaler_X, df, accuracy, n_features = train_lstm(symbol)
    except Exception as e:
        print(f"Error: {e}")
        return {
            "symbol": symbol,
            "predicted_price": 0,
            "current_price": 0,
            "change_percent": 0,
            "suggestion": "ERROR",
            "accuracy_score": 0,
            "data_source_used": "Failed to train",
            "historical_prices": [],
            "dates": [],
        }
    
    # Prepare features for prediction
    feature_cols = [
        'close', 'returns', 'log_returns', 'high_low_ratio', 'open_close_ratio',
        'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_30',
        'macd', 'macd_signal', 'macd_histogram', 'rsi', 'bb_position',
        'volume_ratio', 'volatility', 'momentum_5', 'momentum_10',
        'close_lag_1', 'close_lag_2', 'returns_lag_1'
    ]
    
    available_cols = [col for col in feature_cols if col in df.columns]
    
    # Get last LOOKBACK days
    last_sequence = df[available_cols].tail(LOOKBACK).values
    last_scaled = scaler_X.transform(last_sequence)
    last_scaled = last_scaled.reshape(1, LOOKBACK, n_features)
    
    # Predict next day's return
    predicted_return = float(model.predict(last_scaled, verbose=0)[0][0])
    current_price = float(df["close"].iloc[-1])
    
    # Convert return to price
    predicted_price = current_price * (1 + predicted_return)
    change_percent = predicted_return * 100
    
    # Generate suggestion
    if change_percent > 1:
        suggestion = "STRONG BUY" if change_percent > 2 else "BUY"
    elif change_percent < -1:
        suggestion = "STRONG SELL" if change_percent < -2 else "SELL"
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
        "data_source_used": "Yahoo Finance (Advanced LSTM)",
        "historical_prices": historical_prices,
        "dates": dates,
    }