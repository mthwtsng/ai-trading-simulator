import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

def fetch_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    df["Returns"] = df["Close"].pct_change()
    return df

def prepare_data(df, lookback=5):
    for i in range(1, lookback + 1):
        df[f"Returns_Lag_{i}"] = df["Returns"].shift(i)
    df["Target"] = (df["Returns"].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    features = [f"Returns_Lag_{i}" for i in range(1, lookback + 1)]
    X = df[features].values
    y = df["Target"].values
    return X, y, df, features


def train_models(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    models = {
        "SVM": SVC(probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42)
    }
    for name, model in models.items():
        model.fit(X_scaled, y_train)
    
    return models, scaler

def generate_signals(df, models, scaler, features):
    X = df[features].values
    X_scaled = scaler.transform(X)
    
    predictions = np.zeros((X.shape[0], len(models)))
    for i, (name, model) in enumerate(models.items()):
        predictions[:, i] = model.predict(X_scaled)
    
    df["Signal"] = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)
    df["Signal"] = df["Signal"].map({1: 1, 0: -1}) 
    return df


def simulate_trading(df, initial_cash=10000):
    cash = initial_cash
    shares = 0
    fee_per_share = 0.003 
    portfolio = []
    for i, row in df.iterrows():
        price = row["Close"]
        signal = row["Signal"]
        
        # buy
        if signal == 1 and cash >= price:
            shares_to_buy = int(cash / (price + fee_per_share))
            total_cost = shares_to_buy * price
            total_fee = shares_to_buy * fee_per_share
            if total_cost + total_fee <= cash:
                shares += shares_to_buy
                cash -= total_cost + total_fee
        
        # sell
        elif signal == -1 and shares > 0:
            total_revenue = shares * price
            total_fee = shares * fee_per_share
            cash += total_revenue - total_fee
            shares = 0

        portfolio.append(cash + shares * price)

    df["Portfolio"] = portfolio
    return df


def plot_results(df, symbol):
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df["Close"], label="Stock Price")
    plt.title(f"{symbol} Stock Price and Trading Signals")
    plt.legend()
    
    plt.subplot(3, 1, 2)
    buy_signals = df[df["Signal"] == 1]["Close"]
    sell_signals = df[df["Signal"] == -1]["Close"]
    plt.plot(df.index, df["Close"], label="Stock Price", alpha=0.5)
    plt.scatter(buy_signals.index, buy_signals, color="green", marker="^", label="Buy Signal")
    plt.scatter(sell_signals.index, sell_signals, color="red", marker="v", label="Sell Signal")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df["Portfolio"], label="Portfolio Value", color="purple")
    plt.title("Portfolio Value")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{symbol}_trading_results.png")
    plt.close()


def main(symbol="NVDA", days=252):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = fetch_stock_data(symbol, start_date, end_date)
    X, y, df, features = prepare_data(df)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    models, scaler = train_models(X_train, y_train)
    
    X_test_scaled = scaler.transform(X_test)
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.2f}")
    
    df = generate_signals(df, models, scaler, features)
    df = simulate_trading(df)
    
    plot_results(df, symbol)
    print(f"Final Portfolio Value: ${df['Portfolio'].iloc[-1]:.2f}")

if __name__ == "__main__":
    main()