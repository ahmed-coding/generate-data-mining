import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
import ta
import os
import time

from advanced_model_fixed2 import AdvancedDQNAgent, EnhancedTradingEnv, add_technical_indicators, train_model, backtest_model, plot_performance, plot_trading_performance

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# symbol = "BTCUSDT"
symbol = "ETHUSDT"
# symbol = "SOLUSDT"

# interval = "1h"
# interval = "30m"
# interval = "15m"
interval = "5m"
# interval = "3m"
# interval = "1m"

version_no = '0'


def load_synthetic_data():
    """
    Load synthetic data for testing
    """
    # train_file = 'data/ETHUSDT_synthetic_train.csv'
    # test_file = 'data/ETHUSDT_synthetic_test.csv'

    data = pd.read_csv(
        f'./input/{symbol}/{symbol}_{interval}_binance_candel.csv')

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data["close_time"] = pd.to_datetime(data["close_time"])

    # data['1h'] = 1 if interval == "1h" else 0
    # data['30m'] = 1 if interval == "30m" else 0
    # data['15m'] = 1 if interval == "15m" else 0
    # data['5m'] = 1 if interval == "5m" else 0
    # data['3m'] = 1 if interval == "3m" else 0
    # data['1m'] = 1 if interval == "1m" else 0

    data.set_index('timestamp', inplace=True)
    # data = data.astype(float)
    data = data.drop(columns=['ignore'], errors='ignore')

    # Convert remaining columns to float32
    data = data.apply(pd.to_numeric, errors='coerce').astype(np.float32)
    # data = data.select_dtypes(exclude=['datetime64']).astype(np.float32)
    # Drop NaN values
    # data.dropna(inplace=True)

    tdata = data.copy()
    # data = data.loc["2020":"2024"]

    
    # if not os.path.exists(train_file) or not os.path.exists(test_file):
    #     raise FileNotFoundError("Synthetic data files not found. Run synthetic_data.py first.")

    # train_data = pd.read_csv(train_file, index_col=0)
    # test_data = pd.read_csv(test_file, index_col=0)
    
    test_data = tdata.loc["2025"]
    train_data = data.loc["2022":"2024"]
    # train_data = data.loc["2025"]


    # # Reset index before saving
    # data.reset_index(drop=True, inplace=True)
    
    # Add technical indicators
    train_data = add_technical_indicators(train_data)
    test_data = add_technical_indicators(test_data)

    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")

    return train_data, test_data


def quick_test(episodes=2, batch_size=64):
    """
    Run a quick test of the model with fewer episodes
    """
    # Load data
    train_data, test_data = load_synthetic_data()
    state_lengh = len(train_data)
    # Create training environment
    train_env = EnhancedTradingEnv(train_data)

    # Get state and action dimensions
    state_size = train_env.observation_space.shape[0]
    action_size = train_env.action_space.n

    print(f"State size: {state_size}, Action size: {action_size}")

    # Create agent
    agent = AdvancedDQNAgent(
        state_size, action_size,state_lengh=state_lengh, model_name=f"{symbol}_{version_no}_{interval}_dqn_trader.keras")

    # Train model with fewer episodes for quick testing
    print("Starting quick training...")
    scores, win_rates, balances = train_model(
        train_env, agent, episodes, batch_size)

    # Plot training performance
    plot_performance(scores, win_rates, balances, episodes)

    # Create testing environment
    test_env = EnhancedTradingEnv(test_data)

    # Backtest model
    print("Starting backtesting...")
    info, trades = backtest_model(test_env, agent)

    # Plot trading performance
    plot_trading_performance(test_data, trades, info)

    # Print final results
    print("\nQuick Test Results:")
    print(f"Final Win Rate: {win_rates[-1]:.2f}%")
    print(f"Final Balance: ${balances[-1]:.2f}")

    print("\nBacktesting Results:")
    print(f"Win Rate: {info['win_rate']*100:.2f}%")
    print(f"Final Balance: ${info['balance']:.2f}")
    print(f"Total Trades: {info['trades']}")
    print(f"Max Drawdown: {info['max_drawdown']*100:.2f}%")

    return info['win_rate'] * 100  # Return win rate percentage


if __name__ == "__main__":
    os.makedirs("dqn_trading_model/", exist_ok=True)
    win_rate = quick_test()
    print(f"\nFinal Win Rate: {win_rate:.2f}%")
