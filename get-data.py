# pip install ccxt pandas ta openpyxl
import ccxt
import pandas as pd
import ta  # Technical Analysis Library
# import openpyxl
from datetime import datetime, timedelta
import numpy as np
import os
# Function to fetch historical data from Binance within a date range


symbols = [
    'BTC/USDT',
    # 'ETH/USDT',
    # 'BNB/USDT',
    # 'SOL/USDT',
    # 'SUI/USDT',
    # 'ADA/USDT',
    # 'XRP/USDT',
    # 'DOT/USDT',
    # 'LINK/USDT',
    # 'LTC/USDT',
    # 'BCH/USDT',
    # 'XLM/USDT',
    # 'AVAX/USDT',
    # 'UNI/USDT',
    # 'AAVE/USDT',
    # 'XTZ/USDT',
    # 'ATOM/USDT',
    # 'VET/USDT',
    # 'ALGO/USDT',
    # 'HBAR/USDT',
    # 'MKR/USDT',
    # 'SNX/USDT',
    # 'YFI/USDT',
    # 'COMP/USDT',
    # 'FTM/USDT',
    # 'LUNA/USDT',
    # 'MATIC/USDT',
    # 'FIL/USDT',
    # 'CHZ/USDT',
    # 'THETA/USDT',
    # 'LRC/USDT',
    # 'ONE/USDT',
    # 'SHIB/USDT',
    # 'QNT/USDT',
    # 'ENJ/USDT',
    # 'FLOW/USDT',
    # 'CAKE/USDT',
    # 'CELO/USDT',
    # 'BAT/USDT',
    # 'REN/USDT',
    # 'LDO/USDT',
    # 'RNDR/USDT',
    # 'XDC/USDT',
    # 'SUSHI/USDT',
    # 'BNT/USDT',
    # 'ZEC/USDT',
    # 'DCR/USDT',
    # 'DGB/USDT',
    # 'NEXO/USDT',
    # 'RVN/USDT',
    # 'KSM/USDT',
]


intervals = [
    '1h',
    # '30m',
    # '15m',
    # '5m',
    # '3m',
    # '1m'
]


def fetch_binance_data(symbol, timeframe, start_date, end_date):
    exchange = ccxt.binance()
    since = exchange.parse8601(start_date)
    end_timestamp = exchange.parse8601(end_date)
    all_ohlcv = []

    while since < end_timestamp:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
        if not ohlcv:
            break  # No more data available
        since = ohlcv[-1][0] + 1  # Move to the next time period
        all_ohlcv.extend(ohlcv)

    df = pd.DataFrame(all_ohlcv, columns=[
                      'timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Ensure the end_date is timezone-naive
    end_date_naive = pd.to_datetime(end_date).tz_localize(None)
    # Filter data up to the end date
    df = df[df['timestamp'] <= end_date_naive]
    return df

# Function to calculate ALL technical indicators


def calculate_all_indicators(df):
    # Trend Indicators
    df['sma_2'] = ta.trend.sma_indicator(df['close'], window=2)  # SMA_2
    df['sma_3'] = ta.trend.sma_indicator(df['close'], window=3)  # SMA_3

    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)

    df['ema_3'] = ta.trend.ema_indicator(df['close'], window=3)
    df['ema_5'] = ta.trend.ema_indicator(df['close'], window=5)

    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['wma'] = ta.trend.wma_indicator(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    df['macd_diff'] = ta.trend.macd_diff(df['close'])
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=14)
    df['dpo'] = ta.trend.dpo(df['close'], window=20)
    df['kst'] = ta.trend.kst(df['close'])
    df['kst_sig'] = ta.trend.kst_sig(df['close'])
    df['ichimoku_conv'] = ta.trend.ichimoku_conversion_line(
        df['high'], df['low'])
    df['ichimoku_base'] = ta.trend.ichimoku_base_line(df['high'], df['low'])
    df['ichimoku_a'] = ta.trend.ichimoku_a(df['high'], df['low'])
    df['ichimoku_b'] = ta.trend.ichimoku_b(df['high'], df['low'])
    df['aroon_up'] = ta.trend.aroon_up(
        df['high'], df['low'], window=14)  # Fixed: Added 'high' and 'low'
    df['aroon_down'] = ta.trend.aroon_down(
        df['high'], df['low'], window=14)  # Fixed: Added 'high' and 'low'
    df['stc'] = ta.trend.stc(df['close'])

    # Momentum Indicators
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['stoch'] = ta.momentum.stoch(
        df['high'], df['low'], df['close'], window=14)
    df['stoch_signal'] = ta.momentum.stoch_signal(
        df['high'], df['low'], df['close'], window=14)
    df['tsi'] = ta.momentum.tsi(df['close'])
    df['uo'] = ta.momentum.ultimate_oscillator(
        df['high'], df['low'], df['close'])
    df['wr'] = ta.momentum.williams_r(
        df['high'], df['low'], df['close'], lbp=14)
    df['ao'] = ta.momentum.awesome_oscillator(df['high'], df['low'])
    df['kama'] = ta.momentum.kama(df['close'])

    # Volatility Indicators
    df['atr'] = ta.volatility.average_true_range(
        df['high'], df['low'], df['close'], window=14)
    df['bb_high'] = ta.volatility.bollinger_hband(
        df['close'], window=20, window_dev=2)
    df['bb_low'] = ta.volatility.bollinger_lband(
        df['close'], window=20, window_dev=2)
    df['bb_mavg'] = ta.volatility.bollinger_mavg(df['close'], window=20)
    df['bb_width'] = ta.volatility.bollinger_wband(
        df['close'], window=20, window_dev=2)
    df['kc_high'] = ta.volatility.keltner_channel_hband(
        df['high'], df['low'], df['close'], window=20)
    df['kc_low'] = ta.volatility.keltner_channel_lband(
        df['high'], df['low'], df['close'], window=20)
    df['dc_high'] = ta.volatility.donchian_channel_hband(
        df['high'], df['low'], df['close'], window=20)  # Fixed: Added 'high' and 'low'
    df['dc_low'] = ta.volatility.donchian_channel_lband(
        df['high'], df['low'], df['close'], window=20)  # Fixed: Added 'high' and 'low'

    # Volume Indicators
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['cmf'] = ta.volume.chaikin_money_flow(
        df['high'], df['low'], df['close'], df['volume'], window=20)
    df['fi'] = ta.volume.force_index(df['close'], df['volume'], window=13)
    # Fixed: Removed redundant 'window' argument
    df['eom'] = ta.volume.ease_of_movement(
        df['high'], df['low'], df['volume'], window=14)
    df['vpt'] = ta.volume.volume_price_trend(df['close'], df['volume'])
    df['nvi'] = ta.volume.negative_volume_index(df['close'], df['volume'])
    df['vwap'] = ta.volume.volume_weighted_average_price(
        df['high'], df['low'], df['close'], df['volume'])

    return df


# Strategy: EMA_2 and EMA_3 Crossover Strategy

def ema_crossover_strategy(df):
    df['signal'] = 0
    # Buy signal when EMA_2 > EMA_5
    df.loc[df['ema_3'] > df['ema_5'], 'signal'] = 1  # Buy signal
    df.loc[df['ema_3'] < df['ema_5'], 'signal'] = - \
        1  # Sell signal when EMA_2 < EMA_5
    df['positions'] = df['signal'].diff()
    return df

# Strategy: SMA_2 and SMA_3 Crossover Strategy


def sma_crossover_strategy(df):
    df['signal'] = 0
    # Buy signal when SMA_2 > SMA_3
    df.loc[df['sma_2'] > df['sma_3'], 'signal'] = 1
    df.loc[df['sma_2'] < df['sma_3'], 'signal'] = - \
        1  # Sell signal when SMA_2 < SMA_3
    df['positions'] = df['signal'].diff()
    return df

# Strategy: RSI-Based Strategy for Long and Short Trades


def rsi_strategy(df, rsi_buy_threshold=30, rsi_sell_threshold=70):
    df['signal'] = 0
    # Buy signal when RSI < 30 (oversold)
    df.loc[df['rsi'] < rsi_buy_threshold, 'signal'] = 1
    df.loc[df['rsi'] > rsi_sell_threshold, 'signal'] = - \
        1  # Sell signal when RSI > 70 (overbought)
    df['positions'] = df['signal'].diff()
    return df

# Backtest the strategy with TP and SL
# rsi = 6 of 21, sma = 12 of 18, ema = 12 of 18
# rsi = 6 of 23, sma = 14 of 26, ema = 13 of 26


def backtest_strategy(df, symbol, interval, tp_percent=5, sl_percent=1):
    trades = []
    in_position = False
    position_type = None  # 'long' or 'short'
    entry_price = 0.0

    for index, row in df.iterrows():
        if row['signal'] == 1 and not in_position:  # Buy signal (long)
            in_position = True
            position_type = 'long'
            entry_price = row['close']
            trade = {
                'Symbol': symbol,
                'Interval': interval,
                'Entry Date': row['timestamp'],
                'Entry Price': entry_price,
                'Exit Date': None,
                'Exit Price': None,
                'Profit/Loss': None,
                'Status': 'Open',
                'Type': position_type,
                **{col: row[col] for col in df.columns if col not in ['signal', 'positions']}
            }
            trades.append(trade)
        elif row['signal'] == -1 and not in_position:  # Sell signal (short)
            in_position = True
            position_type = 'short'
            entry_price = row['close']
            trade = {
                'Symbol': symbol,
                'Interval': interval,
                'Entry Date': row['timestamp'],
                'Entry Price': entry_price,
                'Exit Date': None,
                'Exit Price': None,
                'Profit/Loss': None,
                'Status': 'Open',
                'Type': position_type,
                **{col: row[col] for col in df.columns if col not in ['signal', 'positions']}
            }
            trades.append(trade)

        # Check for TP or SL
        if in_position:
            current_price = row['close']
            top_price = row['high']
            low_price = row['low']
            if position_type == 'long':
                tp_price = entry_price * (1 + tp_percent / 100)  # TP for long
                sl_price = entry_price * (1 - sl_percent / 100)  # SL for long
                # if current_price >= tp_price or current_price <= sl_price:
                if top_price >= tp_price or low_price <= sl_price:

                    in_position = False
                    trades[-1]['Exit Date'] = row['timestamp']
                    trades[-1]['Exit Price'] = current_price
                    trades[-1]['Profit/Loss'] = current_price - \
                        trades[-1]['Entry Price']
                    trades[-1]['Status'] = 'Closed'
                    trades[-1]['Win/Lose'] = 'Win' if trades[-1]['Profit/Loss'] > 0 else 'Lose'
            elif position_type == 'short':
                tp_price = entry_price * (1 - tp_percent / 100)  # TP for short
                sl_price = entry_price * (1 + sl_percent / 100)  # SL for short
                # if current_price <= tp_price or current_price >= sl_price:
                if low_price <= tp_price or top_price >= sl_price:

                    in_position = False
                    trades[-1]['Exit Date'] = row['timestamp']
                    trades[-1]['Exit Price'] = current_price
                    trades[-1]['Profit/Loss'] = trades[-1]['Entry Price'] - \
                        current_price
                    trades[-1]['Status'] = 'Closed'
                    trades[-1]['Win/Lose'] = 'Win' if trades[-1]['Profit/Loss'] > 0 else 'Lose'

    # Handle the last trade if it's still open
    if in_position:
        trades[-1]['Exit Date'] = df['timestamp'].iloc[-1]
        trades[-1]['Exit Price'] = df['close'].iloc[-1]
        if position_type == 'long':
            trades[-1]['Profit/Loss'] = trades[-1]['Exit Price'] - \
                trades[-1]['Entry Price']
        else:
            trades[-1]['Profit/Loss'] = trades[-1]['Entry Price'] - \
                trades[-1]['Exit Price']
        trades[-1]['Status'] = 'Closed'
        trades[-1]['Win/Lose'] = 'Win' if trades[-1]['Profit/Loss'] > 0 else 'Lose'

    return pd.DataFrame(trades)

# Save results to CSV


def save_to_csv(results, filename='binance_rsi_tp_sl_trading_results.csv'):
    if os.path.exists(filename):
        # Load existing data
        existing_data = pd.read_csv(filename)
        # Combine existing data with new results
        updated_data = pd.concat([existing_data, results], ignore_index=True)
        # Remove duplicates based on 'Entry Date' and 'Symbol'
        # updated_data = updated_data.drop_duplicates(subset=['Entry Date', 'Symbol'], keep='last')
        updated_data.to_csv(filename, index=False)
        # print(f"Results updated in existing file: {filename}")
    else:
        # Create new file
        results.to_csv(filename, index=False)
        print(f"Results saved to new file: {filename}")

# Main function


# Main function
def main():
    # Parameters
    # symbol = 'LTC/USDT'  # Trading pair
    # interval = '15m'  # Timeframe (1 hour)

    start_date = '2020-01-01T00:00:00Z'  # Start date in UTC
    # end_date = '2025-03-01T00:00:00Z'  # End date in UTC
    end_date = '2020-02-01T00:00:00Z'  # End date in UTC

    # RSI Strategy Parameters
    rsi_buy_threshold = 30  # Buy when RSI < 30 (oversold)
    rsi_sell_threshold = 70  # Sell when RSI > 70 (overbought)

    # TP and SL Parameters
    tp_percent = 4  # Target profit: 5%
    sl_percent = 2  # Stop loss: 1%

    for symbol in symbols:
        if not os.path.exists(f'{symbol.replace("/","")}'):
            os.makedirs(f'{symbol.replace("/","")}')

        rsi_filename = f'{symbol.replace("/","")}/{symbol.replace("/","")}_rsi_binance_futures_trading_results.csv'
        sma_filename = f'{symbol.replace("/","")}/{symbol.replace("/","")}_sma_binance_futures_trading_results.csv'
        ema_filename = f'{symbol.replace("/","")}/{symbol.replace("/","")}_ema_binance_futures_trading_results.csv'

        # rsi_filename = f'{symbol.replace("/","")}_rsi_binance_futures_trading_results.csv'
        # sma_filename = f'{symbol.replace("/","")}_sma_binance_futures_trading_results.csv'
        # ema_filename = f'{symbol.replace("/","")}_ema_binance_futures_trading_results.csv'

        print(f"Start geting data for {symbol}...")
        for interval in intervals:

            # Fetch data from Binance
            print(f"Fetching data from Binance for {symbol}-{interval}...")
            data = fetch_binance_data(symbol, interval, start_date, end_date)

            if data.empty:
                print(
                    f"No data available for {symbol} between {start_date} and {end_date}.")
                return

            # Calculate ALL indicators
            print("Calculating ALL indicators...")
            data = calculate_all_indicators(data)

            # Generate signals
            print("Generating SMA signals...")
            sma_data = sma_crossover_strategy(data.copy())

            print("Generating EMA signals...")
            ema_data = ema_crossover_strategy(data.copy())

            print("Generating RSI signals...")
            rsi_data = rsi_strategy(
                data.copy(), rsi_buy_threshold, rsi_sell_threshold)

            # Backtest the strategy
            print("Backtesting strategy...")
            rsi_trade_results = backtest_strategy(
                rsi_data, symbol, interval, tp_percent, sl_percent)

            sma_trade_results = backtest_strategy(
                sma_data, symbol, interval, tp_percent, sl_percent)

            ema_trade_results = backtest_strategy(
                ema_data, symbol, interval, tp_percent, sl_percent)
            # Save results to Excel
            print("Saving results to Excel...")
            save_to_csv(rsi_trade_results, filename=rsi_filename)
            save_to_csv(sma_trade_results, filename=sma_filename)
            save_to_csv(ema_trade_results, filename=ema_filename)
            print(
                f"Trading results saved to '{rsi_filename}' and '{sma_filename}'.")


if __name__ == "__main__":
    main()
