import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Global cache
trained_models = {}

LOOKBACK = 20
EPOCHS = 15

def fetch_stock_data(symbol, period="6mo"):
    """Fetch stock data."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    
    if df.empty:
        raise ValueError(f"No data available for {symbol}")
    
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    return df

def prepare_lstm_data(df, lookback=LOOKBACK):
    """Prepare sequences for LSTM."""
    prices = df["close"].values.reshape(-1, 1)
    
    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)
    
    # Create sequences
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
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(25, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def calculate_directional_accuracy(y_true, y_pred):
    """Calculate directional accuracy."""
    if len(y_true) < 2:
        return 50.0
    
    true_changes = np.diff(y_true.flatten())
    pred_changes = np.diff(y_pred.flatten())
    
    if len(true_changes) == 0:
        return 50.0
    
    correct = np.sum((true_changes > 0) == (pred_changes > 0))
    accuracy = (correct / len(true_changes)) * 100
    
    return accuracy

# ============================================================
# METHOD 1: Bagging Ensemble with Multiple LSTM Models
# ============================================================
class BaggingLSTM:
    """Bagging ensemble of multiple LSTM models."""
    
    def __init__(self, n_estimators=5, lookback=LOOKBACK, epochs=EPOCHS):
        self.n_estimators = n_estimators
        self.lookback = lookback
        self.epochs = epochs
        self.models = []
        self.scalers = []
        
    def train(self, X, y, scaler):
        """Train multiple LSTM models on bootstrap samples."""
        print(f"  🎯 Training Bagging Ensemble with {self.n_estimators} LSTM models...")
        
        n_samples = len(X)
        self.models = []
        self.scalers = [scaler]  # Store scaler once
        
        for i in range(self.n_estimators):
            # Bootstrap sampling with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Train LSTM model
            model = build_lstm_model(self.lookback)
            model.fit(X_bootstrap, y_bootstrap, epochs=self.epochs, batch_size=16, verbose=0)
            self.models.append(model)
            
            print(f"    ✅ Trained estimator {i+1}/{self.n_estimators}")
        
        return self
    
    def predict(self, X):
        """Average predictions from all models."""
        predictions = []
        for model in self.models:
            pred = model.predict(X, verbose=0).flatten()
            predictions.append(pred)
        
        # Return mean prediction
        return np.mean(predictions, axis=0)
    
    def predict_single(self, X):
        """Predict single sample with uncertainty."""
        predictions = []
        for model in self.models:
            pred = model.predict(X, verbose=0).flatten()
            predictions.append(pred[0])
        
        return np.mean(predictions), np.std(predictions)

# ============================================================
# METHOD 2: Hybrid LSTM + Random Forest / Gradient Boosting
# ============================================================
def extract_lstm_features(X, lstm_model):
    """Extract features from LSTM intermediate layers."""
    from tensorflow.keras.models import Model
    
    # Create feature extractor (output of second LSTM layer)
    feature_extractor = Model(
        inputs=lstm_model.input,
        outputs=lstm_model.layers[2].output  # Second LSTM layer output
    )
    
    features = feature_extractor.predict(X, verbose=0)
    return features

def train_hybrid_model(symbol):
    """Train LSTM + Gradient Boosting ensemble."""
    symbol = symbol.upper()
    
    print(f"\n{'='*50}")
    print(f"  🚀 Training HYBRID Ensemble for {symbol}")
    print(f"{'='*50}")
    
    # Fetch and prepare data
    df = fetch_stock_data(symbol)
    X, y, scaler = prepare_lstm_data(df)
    
    # Split data
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # First, train base LSTM
    print(f"  🧠 Training base LSTM model...")
    base_lstm = build_lstm_model()
    base_lstm.fit(X_train, y_train, epochs=EPOCHS, batch_size=16, verbose=0)
    
    # Extract features using LSTM
    print(f"  🔍 Extracting LSTM features for ensemble...")
    X_train_features = extract_lstm_features(X_train, base_lstm)
    X_test_features = extract_lstm_features(X_test, base_lstm)
    
    # Train Gradient Boosting on LSTM features
    print(f"  🌲 Training Gradient Boosting on extracted features...")
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_train_features, y_train)
    
    # Make predictions
    y_pred_gb = gb_model.predict(X_test_features)
    y_pred_lstm = base_lstm.predict(X_test, verbose=0).flatten()
    
    # Ensemble: weighted average (70% LSTM, 30% GB)
    y_pred_ensemble = (0.7 * y_pred_lstm) + (0.3 * y_pred_gb)
    
    # Calculate accuracy
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_real = scaler.inverse_transform(y_pred_ensemble.reshape(-1, 1))
    accuracy = calculate_directional_accuracy(y_test_real, y_pred_real)
    mape = mean_absolute_percentage_error(y_test_real, y_pred_real) * 100
    
    print(f"\n✅ Hybrid Ensemble Performance:")
    print(f"  • Directional Accuracy: {accuracy:.1f}%")
    print(f"  • MAPE: {mape:.1f}%")
    print(f"{'='*50}\n")
    
    return base_lstm, gb_model, scaler, df, accuracy

# ============================================================
# METHOD 3: Time Series Cross Validation with Grid Search
# ============================================================
def grid_search_lstm_params(symbol):
    """Grid search for optimal LSTM hyperparameters."""
    symbol = symbol.upper()
    
    print(f"\n{'='*50}")
    print(f"  🔍 Grid Search for {symbol}")
    print(f"{'='*50}")
    
    # Fetch data
    df = fetch_stock_data(symbol)
    X, y, scaler = prepare_lstm_data(df)
    
    # Parameter grid
    param_grid = {
        'lstm_units': [32, 64, 128],
        'dropout_rate': [0.1, 0.2, 0.3],
        'epochs': [10, 15, 20],
        'batch_size': [16, 32]
    }
    
    best_accuracy = 0
    best_params = {}
    results = []
    
    # Time series cross validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    from itertools import product
    
    total_combinations = len(param_grid['lstm_units']) * len(param_grid['dropout_rate']) * len(param_grid['epochs']) * len(param_grid['batch_size'])
    print(f"  📊 Testing {total_combinations} parameter combinations...")
    
    combo_count = 0
    for units, dropout_rate, epochs, batch_size in product(
        param_grid['lstm_units'],
        param_grid['dropout_rate'],
        param_grid['epochs'],
        param_grid['batch_size']
    ):
        combo_count += 1
        print(f"    Testing {combo_count}/{total_combinations}...", end='\r')
        
        fold_accuracies = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Build model with current parameters
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            model = Sequential([
                LSTM(units, return_sequences=True, input_shape=(LOOKBACK, 1)),
                Dropout(dropout_rate),
                LSTM(units // 2, return_sequences=False),
                Dropout(dropout_rate),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            
            # Train
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Evaluate
            y_pred = model.predict(X_val, verbose=0).flatten()
            y_val_real = scaler.inverse_transform(y_val.reshape(-1, 1))
            y_pred_real = scaler.inverse_transform(y_pred.reshape(-1, 1))
            accuracy = calculate_directional_accuracy(y_val_real, y_pred_real)
            fold_accuracies.append(accuracy)
        
        mean_accuracy = np.mean(fold_accuracies)
        results.append((mean_accuracy, units, dropout_rate, epochs, batch_size))
        
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = {
                'lstm_units': units,
                'dropout_rate': dropout_rate,
                'epochs': epochs,
                'batch_size': batch_size
            }
    
    print(f"\n\n✅ Grid Search Complete!")
    print(f"  • Best Accuracy: {best_accuracy:.1f}%")
    print(f"  • Best Parameters: {best_params}")
    print(f"{'='*50}\n")
    
    return best_params, best_accuracy

# ============================================================
# MAIN TRAINING FUNCTION (Choose your method)
# ============================================================

def train_lstm(symbol, ensemble_method='bagging'):
    """
    Train LSTM model with ensemble methods.
    
    Methods:
    - 'standard': Regular LSTM
    - 'bagging': Bagging ensemble of multiple LSTMs
    - 'hybrid': LSTM + Gradient Boosting
    - 'grid_search': Find best parameters first
    """
    symbol = symbol.upper()
    
    cache_key = f"{symbol}_{ensemble_method}"
    if cache_key in trained_models:
        print(f"  🔄 Using cached model for {symbol} ({ensemble_method})")
        return trained_models[cache_key]
    
    print(f"\n{'='*50}")
    print(f"  🚀 Training {ensemble_method.upper()} model for {symbol}")
    print(f"{'='*50}")
    
    # Fetch data
    df = fetch_stock_data(symbol)
    print(f"  ✅ Retrieved {len(df)} days of data")
    
    if len(df) < LOOKBACK + 20:
        raise ValueError(f"Insufficient data for {symbol}")
    
    # Prepare data
    X, y, scaler = prepare_lstm_data(df)
    
    # Split data
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    if ensemble_method == 'standard':
        # Standard LSTM
        print(f"  🧠 Training standard LSTM...")
        model = build_lstm_model()
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=16, validation_split=0.1, verbose=1)
        predictions = model.predict(X_test, verbose=0).flatten()
        result_model = model
        
    elif ensemble_method == 'bagging':
        # Bagging ensemble
        bagging = BaggingLSTM(n_estimators=5)
        bagging.train(X_train, y_train, scaler)
        predictions = bagging.predict(X_test)
        result_model = bagging
        
    elif ensemble_method == 'hybrid':
        # Hybrid LSTM + Gradient Boosting
        base_lstm = build_lstm_model()
        base_lstm.fit(X_train, y_train, epochs=EPOCHS, batch_size=16, verbose=0)
        
        X_train_features = extract_lstm_features(X_train, base_lstm)
        X_test_features = extract_lstm_features(X_test, base_lstm)
        
        gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        gb_model.fit(X_train_features, y_train)
        
        y_pred_lstm = base_lstm.predict(X_test, verbose=0).flatten()
        y_pred_gb = gb_model.predict(X_test_features)
        predictions = (0.7 * y_pred_lstm) + (0.3 * y_pred_gb)
        result_model = (base_lstm, gb_model)
        
    elif ensemble_method == 'grid_search':
        # First run grid search
        best_params, _ = grid_search_lstm_params(symbol)
        
        # Train with best parameters
        print(f"  🧠 Training with optimal parameters...")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        model = Sequential([
            LSTM(best_params['lstm_units'], return_sequences=True, input_shape=(LOOKBACK, 1)),
            Dropout(best_params['dropout_rate']),
            LSTM(best_params['lstm_units'] // 2, return_sequences=False),
            Dropout(best_params['dropout_rate']),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)
        predictions = model.predict(X_test, verbose=0).flatten()
        result_model = model
    
    else:
        raise ValueError(f"Unknown method: {ensemble_method}")
    
    # Calculate accuracy
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_real = scaler.inverse_transform(predictions.reshape(-1, 1))
    accuracy = calculate_directional_accuracy(y_test_real, y_pred_real)
    mape = mean_absolute_percentage_error(y_test_real, y_pred_real) * 100
    
    print(f"\n✅ Model Performance for {symbol} ({ensemble_method}):")
    print(f"  • Directional Accuracy: {accuracy:.1f}%")
    print(f"  • MAPE: {mape:.1f}%")
    print(f"{'='*50}\n")
    
    result = (result_model, scaler, df, accuracy)
    trained_models[cache_key] = result
    return result

def predict_price(symbol, ensemble_method='standard'):
    """Predict next day's price using specified ensemble method."""
    symbol = symbol.upper()
    
    try:
        model, scaler, df, accuracy = train_lstm(symbol, ensemble_method)
    except Exception as e:
        print(f"Error: {e}")
        return {
            "symbol": symbol,
            "predicted_price": 0,
            "current_price": 0,
            "change_percent": 0,
            "suggestion": "ERROR",
            "data_source_used": "Failed to train",
            "historical_prices": [],
            "dates": [],
        }
    
    # Get last LOOKBACK days
    last_sequence = df["close"].values[-LOOKBACK:].reshape(-1, 1)
    last_scaled = scaler.transform(last_sequence)
    last_scaled = last_scaled.reshape(1, LOOKBACK, 1)
    
    # Make prediction based on model type
    if isinstance(model, BaggingLSTM):
        pred_mean, pred_std = model.predict_single(last_scaled)
        predicted_price = float(scaler.inverse_transform([[pred_mean]])[0][0])
    elif isinstance(model, tuple):
        # Hybrid model
        base_lstm, gb_model = model
        # Extract features and predict
        from tensorflow.keras.models import Model
        feature_extractor = Model(inputs=base_lstm.input, outputs=base_lstm.layers[2].output)
        lstm_features = feature_extractor.predict(last_scaled, verbose=0)
        pred_lstm = base_lstm.predict(last_scaled, verbose=0).flatten()[0]
        pred_gb = gb_model.predict(lstm_features)[0]
        predicted_price = float(scaler.inverse_transform([[(0.7 * pred_lstm + 0.3 * pred_gb)]])[0][0])
    else:
        # Standard LSTM
        pred_scaled = model.predict(last_scaled, verbose=0)
        predicted_price = float(scaler.inverse_transform(pred_scaled)[0][0])
    
    current_price = float(df["close"].iloc[-1])
    change_percent = ((predicted_price - current_price) / current_price) * 100
    
    # Generate suggestion
    if change_percent > 1.5:
        suggestion = "BUY"
    elif change_percent < -1.5:
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
        "data_source_used": f"Yahoo Finance (LSTM-{ensemble_method})",
        "historical_prices": historical_prices,
        "dates": dates,
    }