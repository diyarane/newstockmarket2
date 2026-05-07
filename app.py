from flask import Flask, render_template, request, jsonify
import requests
import yfinance as yf
from model import predict_price, trained_models

app = Flask(__name__)

ALPHA_VANTAGE_API_KEY = 'A1AP3WSVCIITGGT8'

# Popular stocks for autocomplete seed list
POPULAR_SYMBOLS = [
    {"symbol": "AAPL",  "name": "Apple Inc."},
    {"symbol": "MSFT",  "name": "Microsoft Corporation"},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "AMZN",  "name": "Amazon.com Inc."},
    {"symbol": "TSLA",  "name": "Tesla Inc."},
    {"symbol": "NVDA",  "name": "NVIDIA Corporation"},
    {"symbol": "META",  "name": "Meta Platforms Inc."},
    {"symbol": "BRK-B", "name": "Berkshire Hathaway"},
    {"symbol": "JPM",   "name": "JPMorgan Chase & Co."},
    {"symbol": "V",     "name": "Visa Inc."},
    {"symbol": "UNH",   "name": "UnitedHealth Group"},
    {"symbol": "JNJ",   "name": "Johnson & Johnson"},
    {"symbol": "WMT",   "name": "Walmart Inc."},
    {"symbol": "XOM",   "name": "Exxon Mobil Corporation"},
    {"symbol": "MA",    "name": "Mastercard Inc."},
    {"symbol": "PG",    "name": "Procter & Gamble"},
    {"symbol": "HD",    "name": "The Home Depot"},
    {"symbol": "CVX",   "name": "Chevron Corporation"},
    {"symbol": "ABBV",  "name": "AbbVie Inc."},
    {"symbol": "KO",    "name": "The Coca-Cola Company"},
    {"symbol": "PEP",   "name": "PepsiCo Inc."},
    {"symbol": "LLY",   "name": "Eli Lilly and Company"},
    {"symbol": "MRK",   "name": "Merck & Co."},
    {"symbol": "BAC",   "name": "Bank of America"},
    {"symbol": "AVGO",  "name": "Broadcom Inc."},
    {"symbol": "DIS",   "name": "The Walt Disney Company"},
    {"symbol": "COST",  "name": "Costco Wholesale"},
    {"symbol": "ADBE",  "name": "Adobe Inc."},
    {"symbol": "NFLX",  "name": "Netflix Inc."},
    {"symbol": "CRM",   "name": "Salesforce Inc."},
    {"symbol": "AMD",   "name": "Advanced Micro Devices"},
    {"symbol": "INTC",  "name": "Intel Corporation"},
    {"symbol": "ORCL",  "name": "Oracle Corporation"},
    {"symbol": "PYPL",  "name": "PayPal Holdings"},
    {"symbol": "QCOM",  "name": "Qualcomm Inc."},
    {"symbol": "IBM",   "name": "International Business Machines"},
    {"symbol": "GE",    "name": "General Electric"},
    {"symbol": "F",     "name": "Ford Motor Company"},
    {"symbol": "GM",    "name": "General Motors"},
    {"symbol": "BA",    "name": "The Boeing Company"},
    {"symbol": "GS",    "name": "Goldman Sachs Group"},
    {"symbol": "MS",    "name": "Morgan Stanley"},
    {"symbol": "UBER",  "name": "Uber Technologies"},
    {"symbol": "LYFT",  "name": "Lyft Inc."},
    {"symbol": "SHOP",  "name": "Shopify Inc."},
    {"symbol": "SQ",    "name": "Block Inc."},
    {"symbol": "TWTR",  "name": "Twitter (X)"},
    {"symbol": "SNAP",  "name": "Snap Inc."},
    {"symbol": "SPOT",  "name": "Spotify Technology"},
    {"symbol": "ZM",    "name": "Zoom Video Communications"},
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    symbol = (data.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400
    try:
        result = predict_price(symbol, ensemble_method='bagging')
        # Remove accuracy_score from the response sent to frontend
        if 'accuracy_score' in result:
            del result['accuracy_score']
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



'''
# For standard LSTM (default)
result = predict_price(symbol, ensemble_method='standard')

# For bagging ensemble (usually 2-3% better accuracy)
result = predict_price(symbol, ensemble_method='bagging')

# For hybrid LSTM + Gradient Boosting (best for trending stocks)
result = predict_price(symbol, ensemble_method='hybrid')

# For grid search optimized (slower but finds best params)
result = predict_price(symbol, ensemble_method='grid_search')
'''

@app.route("/api/stock_data")
def api_stock_data():
    symbol = request.args.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="6mo")
        if hist.empty:
            return jsonify({"error": "No data found"}), 404
        prices = [round(float(p), 4) for p in hist["Close"].values]
        dates  = [str(d.date()) for d in hist.index]
        current = prices[-1] if prices else 0
        return jsonify({"symbol": symbol, "prices": prices, "dates": dates, "current_price": current})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip().upper()
    if len(q) < 1:
        return jsonify([])

    # Filter local seed list
    local_matches = [
        s for s in POPULAR_SYMBOLS
        if q in s["symbol"] or q.lower() in s["name"].lower()
    ][:8]

    if local_matches:
        return jsonify(local_matches)

    # Try Alpha Vantage symbol search
    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=SYMBOL_SEARCH&keywords={q}"
            f"&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        resp = requests.get(url, timeout=8)
        data = resp.json()
        matches = data.get("bestMatches", [])
        results = [
            {"symbol": m["1. symbol"], "name": m["2. name"]}
            for m in matches[:8]
            if m.get("4. region") == "United States"
        ]
        return jsonify(results)
    except Exception:
        return jsonify(local_matches)


if __name__ == "__main__":
    print("🚀 Stock Market Predictor starting on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5001)