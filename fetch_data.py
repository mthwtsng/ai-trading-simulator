import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator
import logging

def fetch_stock_data(symbol, start_date, end_date, interval='1d'):

    logger = logging.getLogger('TradingSim')
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date, interval=interval)
        if df.empty or len(df) < 200:
            raise ValueError(f"Not enough data, 200 points or more required")
        df["Returns"] = df["Close"].pct_change()
        df["Log_Returns"] = np.log(df['Close']/df['Close'].shift(1))
        
        df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df['MACD'] = MACD(close=df['Close'], window_slow=26, window_fast=12).macd()
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
        df['EMA_200'] = EMAIndicator(close=df['Close'], window=200).ema_indicator()
        
        na_counts = df[['RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'EMA_50', 'EMA_200']].isna().sum()
        logger.info("NA counts for custom indicators:")
        for col, count in na_counts.items():
            if count > 0:
                logger.info(f"{col}: {count} NA values")
        
        logger.info(f"Retrieved {len(df)} rows for {symbol}")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise

if __name__ == "__main__":
    symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2024-12-31"

    df = fetch_stock_data(symbol, start_date, end_date)
    print(df.tail())