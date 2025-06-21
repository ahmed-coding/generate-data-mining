import os
import pandas as pd
import numpy as np
from advanced_model_fixed2_copy import AdvancedDQNAgent, EnhancedTradingEnv, add_technical_indicators

def load_data():
    data_path = os.path.join('pretrain_data', 'simple_strategy_training_data.csv')
    df = pd.read_csv(data_path)
    df = add_technical_indicators(df)
    return df

def train_agent(episodes=10, pretrain=False):
    df = load_data()
    env = EnhancedTradingEnv(df)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = AdvancedDQNAgent(state_size, action_size)

    # Normal training (random actions, no DQN learning)
    if pretrain:
        print("Pretraining with random actions...")
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = env.action_space.sample()  # Random action
                next_state, reward, done, info = env.step(action)
                state = next_state
                total_reward += reward
            print(f"[Pretrain] Episode {e+1}/{episodes} - Total Reward: {total_reward:.2f}")
        agent.save("pretrain_random_model.keras")
        print("Pretraining complete. Model saved.")
        return

    # DQN training
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()
        print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward:.2f}")
        if (e+1) % 5 == 0:
            agent.save(f"dqn_model_ep{e+1}.keras")

if __name__ == "__main__":
    # First, pretrain with random actions
    train_agent(episodes=5, pretrain=True)
    # Then, train with DQN
    train_agent(episodes=10, pretrain=False)

