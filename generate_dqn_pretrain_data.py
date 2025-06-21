# generate_dqn_pretrain_data.py
"""
Generate (state, action, reward, next_state, done) tuples from classic strategies for DQN pretraining.
"""
import pandas as pd
import numpy as np
import ta
import os
import importlib.util
import sys

# Dynamically import get-data.py as a module
spec = importlib.util.spec_from_file_location("get_data", "get-data.py")
get_data = importlib.util.module_from_spec(spec)
sys.modules["get_data"] = get_data
spec.loader.exec_module(get_data)

# Choose your strategy here:
STRATEGY_FN = get_data.rsi_strategy  # or get_data.sma_crossover_strategy, get_data.ema_crossover_strategy
STRATEGY_NAME = 'rsi'  # 'sma', 'ema', etc.

# Data parameters
SYMBOL = 'BTC/USDT'
INTERVAL = '1h'
DATA_PATH = f'./input/{SYMBOL.replace("/","")}/{SYMBOL.replace("/","")}_{INTERVAL}_binance_candel.csv'

# DQN parameters
WINDOW_SIZE = 20  # Number of bars in state

# Output file
OUTPUT_FILE = f'dqn_pretrain_{STRATEGY_NAME}_{SYMBOL.replace("/","")}_{INTERVAL}.npz'

def build_state(df, idx, window=WINDOW_SIZE):
    # Use only numeric columns for state
    numeric_df = df.select_dtypes(include=[np.number])
    start = max(0, idx - window + 1)
    window_df = numeric_df.iloc[start:idx+1]
    arr = window_df.values
    # Normalize each feature
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0) + 1e-8
    normed = (arr - mean) / std
    return normed.flatten()

def main():
    # Load and prepare data
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = get_data.calculate_all_indicators(df)
    df = STRATEGY_FN(df)
    df = df.dropna().reset_index(drop=True)

    # Build DQN tuples
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for i in range(WINDOW_SIZE, len(df)-1):
        state = build_state(df, i-1)
        next_state = build_state(df, i)
        action = int(df.loc[i, 'signal']) + 1  # Map -1,0,1 to 0,1,2 (DQN expects 0-based)
        reward = df.loc[i, 'close'] - df.loc[i-1, 'close'] if action != 1 else 0  # Simple reward: price diff if not hold
        done = (i == len(df)-2)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)

    # Save as npz
    np.savez(OUTPUT_FILE,
             states=np.array(states),
             actions=np.array(actions),
             rewards=np.array(rewards),
             next_states=np.array(next_states),
             dones=np.array(dones))
    print(f"Saved DQN pretrain data to {OUTPUT_FILE}.")

    # Save as CSV for inspection
    csv_data = []
    for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
        csv_data.append({
            'state': s.tolist(),
            'action': a,
            'reward': r,
            'next_state': ns.tolist(),
            'done': d
        })
    csv_df = pd.DataFrame(csv_data)
    csv_output_file = OUTPUT_FILE.replace('.npz', '.csv')
    csv_df.to_csv(csv_output_file, index=False)
    print(f"Saved DQN pretrain data as CSV to {csv_output_file}.")

    # Also add action, reward, done as columns to the original DataFrame for reference
    df_out = df.copy()
    # Fill with NaN or default, then update only the rows used for DQN tuples
    df_out['dqn_action'] = np.nan
    df_out['dqn_reward'] = np.nan
    df_out['dqn_done'] = np.nan
    for idx, (a, r, d) in enumerate(zip(actions, rewards, dones), start=WINDOW_SIZE):
        df_out.at[idx, 'dqn_action'] = a
        df_out.at[idx, 'dqn_reward'] = r
        df_out.at[idx, 'dqn_done'] = d
    df_out_file = OUTPUT_FILE.replace('.npz', '_with_dqn_cols.csv')
    df_out.to_csv(df_out_file, index=False)
    print(f"Saved DataFrame with DQN columns to {df_out_file}.")

if __name__ == "__main__":
    main()
