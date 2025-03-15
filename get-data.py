# pip install ccxt pandas ta openpyxl
import ccxt
import pandas as pd
import ta  # Technical Analysis Library
import openpyxl
from datetime import datetime, timedelta

# Function to fetch historical data from Binance within a date range
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

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Ensure the end_date is timezone-naive
    end_date_naive = pd.to_datetime(end_date).tz_localize(None)
    df = df[df['timestamp'] <= end_date_naive]  # Filter data up to the end date
    return df

# Function to calculate ALL technical indicators
def calculate_all_indicators(df):
    # Trend Indicators
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
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
    df['ichimoku_conv'] = ta.trend.ichimoku_conversion_line(df['high'], df['low'])
    df['ichimoku_base'] = ta.trend.ichimoku_base_line(df['high'], df['low'])
    df['ichimoku_a'] = ta.trend.ichimoku_a(df['high'], df['low'])
    df['ichimoku_b'] = ta.trend.ichimoku_b(df['high'], df['low'])
    df['aroon_up'] = ta.trend.aroon_up(df['close'], window=14)
    df['aroon_down'] = ta.trend.aroon_down(df['close'], window=14)
    df['stc'] = ta.trend.stc(df['close'])

    # Momentum Indicators
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
    df['stoch_signal'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14)
    df['tsi'] = ta.momentum.tsi(df['close'])
    df['uo'] = ta.momentum.ultimate_oscillator(df['high'], df['low'], df['close'])
    df['wr'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
    df['ao'] = ta.momentum.awesome_oscillator(df['high'], df['low'])
    df['kama'] = ta.momentum.kama(df['close'])

    # Volatility Indicators
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['bb_high'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
    df['bb_low'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
    df['bb_mavg'] = ta.volatility.bollinger_mavg(df['close'], window=20)
    df['bb_width'] = ta.volatility.bollinger_wband(df['close'], window=20, window_dev=2)
    df['kc_high'] = ta.volatility.keltner_channel_hband(df['high'], df['low'], df['close'], window=20)
    df['kc_low'] = ta.volatility.keltner_channel_lband(df['high'], df['low'], df['close'], window=20)
    df['dc_high'] = ta.volatility.donchian_channel_hband(df['close'], window=20)
    df['dc_low'] = ta.volatility.donchian_channel_lband(df['close'], window=20)

    # Volume Indicators
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=20)
    df['fi'] = ta.volume.force_index(df['close'], df['volume'], window=13)
    df['eom'] = ta.volume.ease_of_movement(df['high'], df['low'], df['close'], df['volume'], window=14)
    df['vpt'] = ta.volume.volume_price_trend(df['close'], df['volume'])
    df['nvi'] = ta.volume.negative_volume_index(df['close'], df['volume'])
    df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])

    return df

# Strategy: SMA Crossover for Long and Short Trades
def sma_crossover_strategy(df):
    df['signal'] = 0
    df['signal'][20:] = np.where(
        df['sma_20'][20:] > df['sma_50'][20:], 1, -1  # 1 for Buy, -1 for Sell
    )
    df['positions'] = df['signal'].diff()
    return df

# Backtest the strategy
def backtest_strategy(df, symbol, interval):
    trades = []
    in_position = False
    position_type = None  # 'long' or 'short'
    entry_price = 0.0

    for index, row in df.iterrows():
        if row['positions'] == 1 and not in_position:  # Buy signal (long)
            in_position = True
            position_type = 'long'
            entry_price = row['close']
            trade = {
                'Symbol': symbol,
                'Interval': interval,
                'Entry Date': index,
                'Entry Price': entry_price,
                'Exit Date': None,
                'Exit Price': None,
                'Profit/Loss': None,
                'Status': 'Open',
                'Type': position_type,
                **{col: row[col] for col in df.columns if col not in ['signal', 'positions']}
            }
            trades.append(trade)
        elif row['positions'] == -1 and not in_position:  # Sell signal (short)
            in_position = True
            position_type = 'short'
            entry_price = row['close']
            trade = {
                'Symbol': symbol,
                'Interval': interval,
                'Entry Date': index,
                'Entry Price': entry_price,
                'Exit Date': None,
                'Exit Price': None,
                'Profit/Loss': None,
                'Status': 'Open',
                'Type': position_type,
                **{col: row[col] for col in df.columns if col not in ['signal', 'positions']}
            }
            trades.append(trade)
        elif (row['positions'] == -1 and in_position and position_type == 'long') or (row['positions'] == 1 and in_position and position_type == 'short'):  # Close position
            in_position = False
            exit_price = row['close']
            trades[-1]['Exit Date'] = index
            trades[-1]['Exit Price'] = exit_price
            if position_type == 'long':
                trades[-1]['Profit/Loss'] = exit_price - trades[-1]['Entry Price']
            else:
                trades[-1]['Profit/Loss'] = trades[-1]['Entry Price'] - exit_price
            trades[-1]['Status'] = 'Closed'
            trades[-1]['Win/Lose'] = 'Win' if trades[-1]['Profit/Loss'] > 0 else 'Lose'

    # Handle the last trade if it's still open
    if in_position:
        trades[-1]['Exit Date'] = df.index[-1]
        trades[-1]['Exit Price'] = df['close'].iloc[-1]
        if position_type == 'long':
            trades[-1]['Profit/Loss'] = trades[-1]['Exit Price'] - trades[-1]['Entry Price']
        else:
            trades[-1]['Profit/Loss'] = trades[-1]['Entry Price'] - trades[-1]['Exit Price']
        trades[-1]['Status'] = 'Closed'
        trades[-1]['Win/Lose'] = 'Win' if trades[-1]['Profit/Loss'] > 0 else 'Lose'

    return pd.DataFrame(trades)

# Save results to Excel
def save_to_excel(results, filename='binance_futures_trading_results.xlsx'):
    results.to_excel(filename, index=False)

# Main function
def main():
    # Parameters
    symbol = 'BTC/USDT'  # Trading pair
    interval = '1h'  # Timeframe (1 hour)
    start_date = '2022-01-01T00:00:00Z'  # Start date in UTC
    end_date = '2023-01-01T00:00:00Z'  # End date in UTC

    # Fetch data from Binance
    print("Fetching data from Binance...")
    data = fetch_binance_data(symbol, interval, start_date, end_date)

    if data.empty:
        print(f"No data available for {symbol} between {start_date} and {end_date}.")
        return

    # Calculate ALL indicators
    print("Calculating ALL indicators...")
    data = calculate_all_indicators(data)

    # Generate signals
    print("Generating signals...")
    data = sma_crossover_strategy(data)

    # Backtest the strategy
    print("Backtesting strategy...")
    trade_results = backtest_strategy(data, symbol, interval)

    # Save results to Excel
    print("Saving results to Excel...")
    save_to_excel(trade_results, filename='binance_futures_trading_results.xlsx')
    print("Trading results saved to 'binance_futures_trading_results.xlsx'.")

if __name__ == "__main__":
    main()