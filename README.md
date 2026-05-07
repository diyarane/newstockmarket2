# StockSeer — LSTM Stock Price Predictor

A professional stock market prediction web app using Flask + TensorFlow LSTM neural network.

## Features
- **LSTM Neural Network** — 64→32 units, 60-day lookback, 30 epochs
- **Dual data source** — Alpha Vantage (primary) with Yahoo Finance fallback
- **R² accuracy** — printed in terminal AND displayed in the web UI
- **Model caching** — trains once per symbol, instant on repeat queries
- **Interactive Chart.js** chart with historical + predicted point
- **Auto-refresh** — 1 / 5 / 15 minute intervals
- **Autocomplete** — fuzzy symbol search powered by Alpha Vantage + local seed list

## Quick Start

### 1. Create & activate a virtual environment (recommended)
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ **TensorFlow note:**
> - macOS M1/M2: `pip install tensorflow-macos tensorflow-metal` instead of `tensorflow`
> - CPU-only machines are fine; training takes ~30-60s per symbol
> - GPU (CUDA) will train 5-10× faster

### 3. Run the Flask app
```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## Usage
1. Type a stock ticker in the search box (e.g. `AAPL`, `TSLA`, `NVDA`)
2. Select from the autocomplete dropdown or press **Enter**
3. Click **Predict** — the model trains on first run (~30-60 s), then caches
4. Results appear:
   - Current price & LSTM-predicted next close
   - Change % and BUY / HOLD / SELL recommendation
   - R² model accuracy (also printed in the terminal)
   - Which data source was used (Alpha Vantage or Yahoo Finance fallback)
   - Interactive price chart with the predicted point plotted

### Terminal output example
```
==================================================
  🚀 Training LSTM model for AAPL
==================================================
  ✅ Alpha Vantage returned 1547 rows for AAPL

✅ Model accuracy for AAPL: 91.4% using Alpha Vantage
==================================================
```

---

## Project Structure
```
stock_predictor/
├── app.py               # Flask backend (routes)
├── model.py             # LSTM logic (fetch, train, predict)
├── requirements.txt
├── templates/
│   └── index.html       # Frontend UI
└── static/
    ├── style.css        # Dark theme
    └── script.js        # Chart.js + API calls
```

## API Endpoints
| Endpoint | Method | Params | Description |
|---|---|---|---|
| `/` | GET | — | Serves the main HTML page |
| `/api/predict` | POST | `{"symbol": "AAPL"}` | Returns prediction JSON |
| `/api/stock_data` | GET | `?symbol=AAPL` | Returns 6-month historical data |
| `/api/search` | GET | `?q=AAP` | Returns autocomplete suggestions |

## Configuration
Edit the top of `model.py` to change:
```python
LOOKBACK = 60   # days of history fed to LSTM
EPOCHS   = 30   # training iterations
```

## Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError: tensorflow` | Install with `pip install tensorflow` or `tensorflow-macos` on Mac M-series |
| Alpha Vantage "rate limit" | Free tier = 25 requests/day. The app automatically falls back to Yahoo Finance. |
| Slow first prediction | Normal — LSTM trains from scratch on the first call (~30-60 s). Subsequent calls use the cache. |
| `yfinance` returns empty data | Some non-US symbols may not be on Yahoo Finance. Try the US ticker directly. |

---

*For educational purposes only. Not financial advice.*
