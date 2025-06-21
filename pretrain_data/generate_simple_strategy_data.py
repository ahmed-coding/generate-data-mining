import os
import pandas as pd

def generate_sma_crossover_signals(df, short_window=7, long_window=25):
    df = df.copy()
    df['sma_short'] = df['close'].rolling(window=short_window).mean()
    df['sma_long'] = df['close'].rolling(window=long_window).mean()
    df['signal'] = 0
    df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1  # Buy
    df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1 # Sell
    df['signal'] = df['signal'].shift(1).fillna(0)  # Avoid lookahead bias
    return df

def main():
    input_path = os.path.join('input', 'BTCUSDT', 'BTCUSDT_1h_binance_candel.csv')
    output_dir = os.path.join('pretrain_data')
    output_path = os.path.join(output_dir, 'simple_strategy_training_data.csv')

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    if 'close' not in df.columns:
        raise ValueError('Input CSV must have a "close" column.')
    df = generate_sma_crossover_signals(df)
    df.dropna(inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
