import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator
import logging

def fetch_stock_data(symbol, start_date, end_date, interval='1d'):

    logger = logging.getLogger('Fetch')
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

def prepare_data(df, lookback_days=[3, 5, 10]):
    """
    Prepare features and target for model training.
    """
    logger = logging.getLogger('prepare')
    logger.info("Starting feature engineering")

    df = df.copy()
    for days in lookback_days:
        df[f'Returns_{days}d'] = df['Close'].pct_change(days)
        df[f'Volatility_{days}d'] = df['Log_Returns'].rolling(days).std()
        df[f'MA_{days}d'] = df['Close'].rolling(days).mean()

    df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
    df['Price_Above_EMA50'] = (df['Close'] > df['EMA_50']).astype(int)
    df['Price_Above_EMA200'] = (df['Close'] > df['EMA_200']).astype(int)
    df['EMA50_Above_EMA200'] = (df['EMA_50'] > df['EMA_200']).astype(int)

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    features = (
        [f'Returns_{d}d' for d in lookback_days] +
        [f'Volatility_{d}d' for d in lookback_days] +
        [f'MA_{d}d' for d in lookback_days] +
        ['RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'EMA_50', 'EMA_200',
         'Price_Above_EMA50', 'Price_Above_EMA200', 'EMA50_Above_EMA200']
    )
    critical_columns = ['Close', 'Returns', 'Log_Returns', 'Target'] + features

    original_len = len(df)
    df = df[critical_columns].dropna()
    dropped = original_len - len(df)
    logger.info(f"Dropped {dropped} rows due to NA values in critical columns")

    if len(df) < 10:
        raise ValueError("Insufficient data. Less than 10 rows of data left")

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        logger.error(f"Missing features: {missing_features}")
        raise ValueError(f"Missing features in DataFrame: {missing_features}")

    X = df[features].values
    y = df['Target'].values

    logger.info(f"Prepared {len(X)} samples with {len(features)} features")
    return X, y, df, features