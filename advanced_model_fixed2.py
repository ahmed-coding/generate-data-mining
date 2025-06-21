import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate, BatchNormalization, Add, Lambda
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import gym
from gym import spaces
import ta
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class AdvancedDQNAgent:
    def __init__(self, state_size, action_size, state_lengh = 1000000, model_name="dqn_model.keras"):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=state_lengh)  # Increased memory size
        self.gamma = 0.90    # Higher discount rate for long-term rewards
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9975  # Slower decay for better exploration
        self.learning_rate = 0.0005  # Lower learning rate for stability
        self.update_target_frequency = 10  # Update target network more frequently
        self.batch_size = 256  # Larger batch size
        self.model_name = model_name
        
        # PER (Prioritized Experience Replay) parameters
        self.use_per = True
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling weight
        self.beta_increment = 0.001
        self.priorities = deque(maxlen=state_lengh)
        
        # Force CPU-only mode
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        self.device = '/GPU:0'
        
        # Create main model and target model
        self.model = self._build_advanced_model()
        self.target_model = self._build_advanced_model()
        self.update_target_model()
        
    def init_model(self):
        pass
    def _build_advanced_model(self):
        """Build an advanced DQN model with deeper architecture and dueling architecture"""
        # Market data input
        with tf.device(self.device):
            market_input = Input(shape=(self.state_size,))
            
            # First processing block
            x = Dense(256, activation='relu')(market_input)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # Second processing block
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # Third processing block with residual connection
            y = Dense(128, activation='relu')(x)
            y = BatchNormalization()(y)
            y = Dropout(0.2)(y)
            x = Add()([x, y])  # Residual connection using Keras Add layer
            
            # Final processing
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            
            # Dueling DQN architecture
            # Value stream
            value_stream = Dense(32, activation='relu')(x)
            value = Dense(1)(value_stream)
            
            # Advantage stream
            advantage_stream = Dense(32, activation='relu')(x)
            advantage = Dense(self.action_size)(advantage_stream)
            
            # Combine value and advantage streams using Keras layers
            # Subtract mean of advantage from advantage values
            mean_advantage = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage)
            advantage_subtract = Lambda(lambda inputs: inputs[0] - inputs[1])([advantage, mean_advantage])
            
            # Add value and normalized advantage
            outputs = Add()([value, advantage_subtract])
            
            # Create model
            model = Model(inputs=market_input, outputs=outputs)
            model.compile(loss='huber', optimizer=Adam(learning_rate=self.learning_rate))  # Changed from 'huber_loss' to 'huber'
            
            return model
    
    def update_target_model(self):
        """Update target model weights with current model weights"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory with priority"""
        # Calculate initial priority (TD error or constant for new experiences)
        if len(self.memory) > 0:
            max_priority = max(self.priorities)
        else:
            max_priority = 1.0
            
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def act(self, state, training=True):
        """Choose action based on epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with tf.device(self.device):
            act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        """Train model with experiences from memory using PER"""
        if len(self.memory) < self.batch_size:
            return
        
        # PER sampling
        if self.use_per:
            # Convert priorities to probabilities
            priorities = np.array(self.priorities)
            probabilities = priorities ** self.alpha
            probabilities = probabilities / np.sum(probabilities)
            
            # Sample batch based on priorities
            indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
            
            # Calculate importance sampling weights
            weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
            weights = weights / np.max(weights)  # Normalize weights
            
            # Increment beta for annealing
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            # Get experiences from sampled indices
            minibatch = [self.memory[idx] for idx in indices]
        else:
            # Regular random sampling
            minibatch = random.sample(self.memory, self.batch_size)
            indices = None
            weights = np.ones(self.batch_size)
        
        # Extract data from minibatch
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        with tf.device(self.device):
            # Calculate target Q values using Double DQN approach
            # 1. Get actions from main model
            next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
            
            # 2. Get Q values from target model for those actions
            target_q_values = self.target_model.predict(next_states, verbose=0)
            
            # 3. Calculate target values
            targets = rewards + self.gamma * target_q_values[np.arange(self.batch_size), next_actions] * (1 - dones)
            
            # 4. Get current Q values from main model
            target_f = self.model.predict(states, verbose=0)
            
            # 5. Calculate TD errors for updating priorities
            td_errors = np.abs(targets - target_f[np.arange(self.batch_size), actions])
            
            # 6. Update target values for actions taken
            target_f[np.arange(self.batch_size), actions] = targets
            
            # 7. Train the model with importance sampling weights
            self.model.fit(states, target_f, epochs=10, verbose=1, sample_weight=weights)
            print('-'*20)
        # 8. Update priorities in memory
        if self.use_per and indices is not None:
            for i, idx in enumerate(indices):
                if idx < len(self.priorities):  # Safety check
                    self.priorities[idx] = td_errors[i] + 1e-5  # Add small constant to avoid zero priority
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    # def load(self, name = 'BTCUSDT_4_dqn_trader.h5'):
    def load(self, name=None):
        """Load model weights from file"""
        # if not name:
        #     return
        if os.path.exists(self.model_name):
            self.model.load_weights(name)
            self.update_target_model()
        
    def save(self,name =None ):
        """Save model weights to file"""
        # self.model.save_weights(name)
        self.model.save(name or self.model_name )
        


class EnhancedTradingEnv(gym.Env):
    """
    Enhanced Trading Environment that supports both long and short positions
    with advanced features and market state tracking.
    Modified reward structure for frequent, small, accurate profits.
    """
    def __init__(self, data, initial_balance=1000, commission=0.001, window_size=20):
        super(EnhancedTradingEnv, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        
        self.action_space = spaces.Discrete(3) # 0=hold, 1=buy/long, 2=sell/short
        
        # Observation space (OHLCV + technical indicators + position info + market state)
        # Ensure data.columns is available and correct
        if data is None or data.empty:
            # Handle case with no data, perhaps raise error or use a default shape
            # For now, assuming data is always provided and valid as per original structure
            obs_shape = (window_size + 4,) # Fallback, adjust as needed
        else:
            obs_shape = (len(self.data.columns) + 4,)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        
        self.market_trend = 0  # 0=neutral, 1=uptrend, -1=downtrend
        self.volatility = 0
        
        self.reset()
        
    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = None  # None=no position, "long"=long position, "short"=short position
        self.entry_price = 0.0
        # self.reward = 0 # Step reward, initialized in step()
        self.total_trades = 0
        self.winning_trades = 0
        self.list_trades = []
        self.unrealized_pnl = 0.0
        self.equity = self.balance
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        self.current_price = 0.0
        self._update_market_state()
        return self._next_observation()
    
    def _update_market_state(self):
        if self.current_step < self.window_size or self.current_step >= len(self.data):
            self.market_trend = 0
            self.volatility = 0
            return
            
        price_window = self.data.iloc[max(0, self.current_step - self.window_size):self.current_step]["close"].values
        if len(price_window) == 0:
            self.market_trend = 0
            self.volatility = 0
            return
            
        sma = np.mean(price_window)
        current_price = price_window[-1]
        
        if sma == 0: # Avoid division by zero for volatility, and handle empty/zero price window
            self.market_trend = 0
            self.volatility = 0
            return

        if current_price > sma * 1.01: self.market_trend = 1
        elif current_price < sma * 0.99: self.market_trend = -1
        else: self.market_trend = 0
            
        self.volatility = np.std(price_window) / sma
    
    def _next_observation(self):
        if self.current_step >= len(self.data):
            # Handle edge case: if current_step is out of bounds, return a zero observation or last valid
            # This might happen if called after done=True. For simplicity, return zeros.
            # The shape should match self.observation_space.shape
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        obs_data = self.data.iloc[self.current_step].values
        
        mean_obs = np.mean(obs_data)
        std_obs = np.std(obs_data)
        obs_norm = (obs_data - mean_obs) / (std_obs + 1e-8)
        
        pos_info = [1.0 if self.position == "long" else 0.0, 
                   1.0 if self.position == "short" else 0.0]
        market_info = [float(self.market_trend), float(self.volatility)]
        
        return np.concatenate((obs_norm, pos_info, market_info)).astype(np.float32)

    def _close_position(self, price_at_close):
        realized_profit = 0.0
        if self.position == "long":
            realized_profit = price_at_close * (1 - self.commission) - self.entry_price
        elif self.position == "short":
            realized_profit = self.entry_price - price_at_close * (1 + self.commission)
        else: 
            return 0.0 
            
        self.balance += realized_profit
        self.total_trades += 1
        if realized_profit > 0:
            self.winning_trades += 1
            
        self.position = None
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        
        return realized_profit

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        if done: # If already at the end, no further actions or rewards, return last observation
            obs = self._next_observation()
            # Ensure equity and balance are updated if a position was open till the end
            if self.position is not None:
                 price = self.data.iloc[self.current_step -1]["close"] # Use last available price
                 self.current_price = price
                 if self.position == "long":
                     self.unrealized_pnl = price * (1-self.commission) - self.entry_price
                 else:
                     self.unrealized_pnl = self.entry_price - price * (1+self.commission)
                 self.equity = self.balance + self.unrealized_pnl
            else:
                self.equity = self.balance
            info = self._get_info()
            return obs, 0.0, done, info

        self._update_market_state()
        price = self.data.iloc[self.current_step]["close"]
        self.current_price = price
        step_reward = 0.0

        entry_price_of_potential_closing_trade = self.entry_price
        closed_profit_from_action = 0.0
        position_closed_by_action_this_step = False

        if action == 1: # Action: Buy/Long
            if self.position == "short":
                closed_profit_from_action = self._close_position(price)
                position_closed_by_action_this_step = True
            if self.position is None:
                self.position = "long"
                self.entry_price = price * (1 + self.commission)
                self.list_trades.append((self.current_step, price, "BUY"))
        elif action == 2: # Action: Sell/Short
            if self.position == "long":
                closed_profit_from_action = self._close_position(price)
                position_closed_by_action_this_step = True
            if self.position is None:
                self.position = "short"
                self.entry_price = price * (1 - self.commission)
                self.list_trades.append((self.current_step, price, "SELL"))

        if position_closed_by_action_this_step:
            realized_profit_pct = closed_profit_from_action / entry_price_of_potential_closing_trade if entry_price_of_potential_closing_trade != 0 else 0
            if closed_profit_from_action > 0.0001: # Profit threshold
                step_reward += 1.5
            elif closed_profit_from_action < -0.0001: # Loss threshold
                step_reward -= 10.0
                step_reward -= abs(realized_profit_pct) * 20

        closed_profit_from_sltp = 0.0
        entry_price_for_sltp_check = self.entry_price
        if self.position is not None and not position_closed_by_action_this_step:
            sl_multiplier = max(1.0, 1.5 * self.volatility)
            tp_multiplier = max(1.5, 2.0 * self.volatility)
            sl_triggered, tp_triggered = False, False
            if self.position == "long":
                sl_price = entry_price_for_sltp_check * (1 - 0.015 * sl_multiplier)
                tp_price = entry_price_for_sltp_check * (1 + 0.03 * tp_multiplier)
                sl_triggered = price <= sl_price
                tp_triggered = price >= tp_price
            else: # Short
                sl_price = entry_price_for_sltp_check * (1 + 0.015 * sl_multiplier)
                tp_price = entry_price_for_sltp_check * (1 - 0.03 * tp_multiplier)
                sl_triggered = price >= sl_price
                tp_triggered = price <= tp_price
            if sl_triggered or tp_triggered:
                closed_profit_from_sltp = self._close_position(price)
                realized_profit_pct_sltp = closed_profit_from_sltp / entry_price_for_sltp_check if entry_price_for_sltp_check != 0 else 0
                if closed_profit_from_sltp > 0.0001:
                    step_reward += 1.5
                elif closed_profit_from_sltp < -0.0001:
                    step_reward -= 10.0
                    step_reward -= abs(realized_profit_pct_sltp) * 20
        
        if self.position is not None:
            current_unrealized_profit = 0.0
            if self.position == "long":
                current_unrealized_profit = price * (1 - self.commission) - self.entry_price
            else: # Short
                current_unrealized_profit = self.entry_price - price * (1 + self.commission)
            self.unrealized_pnl = current_unrealized_profit
            self.equity = self.balance + self.unrealized_pnl
            if current_unrealized_profit > 0:
                step_reward += 0.1
            if (self.position == "long" and self.market_trend == 1) or (self.position == "short" and self.market_trend == -1):
                step_reward += 0.05
        else:
            self.unrealized_pnl = 0.0
            self.equity = self.balance

        current_peak = max(self.peak_balance, self.initial_balance)
        if self.equity > current_peak: # Update peak_balance based on equity after trade
             self.peak_balance = self.equity
             current_peak = self.equity # update current_peak for drawdown calc this step

        if current_peak > 0:
            drawdown = (current_peak - self.equity) / current_peak
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        else:
            self.max_drawdown = 0 if self.equity >=0 else 1.0

        obs = self._next_observation()
        info = self._get_info()
        return obs, step_reward, done, info

    def _get_info(self):
        
        return {
            'balance': self.balance,
            'equity': self.equity,
            'price': self.current_price,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0,
            'position': self.position,
            'entry_price': self.entry_price,
            'unrealized_pnl': self.unrealized_pnl,
            'max_drawdown': self.max_drawdown,
            'market_trend': self.market_trend,
            'volatility': self.volatility,
            'trades': self.total_trades,
        }



def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Add trend indicators
    df['sma_7'] = ta.trend.sma_indicator(df['close'], window=7)
    df['sma_25'] = ta.trend.sma_indicator(df['close'], window=25)
    df['sma_99'] = ta.trend.sma_indicator(df['close'], window=99)
    
    df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
    
    # Add MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Add RSI
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
    
    # Add Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
    
    # Add ATR (Average True Range) for volatility
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    # Add Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Add ADX (Average Directional Index)
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    
    # Add OBV (On-Balance Volume)
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    # Add price rate of change
    df['roc_5'] = ta.momentum.roc(df['close'], window=5)
    df['roc_21'] = ta.momentum.roc(df['close'], window=21)
    
    # Add Williams %R
    df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
    
    # Add Ichimoku Cloud components
    ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    
    # Add CCI (Commodity Channel Index)
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
    
    # Drop NaN values
    df = df.dropna()
    
    return df


version = '1'

def train_model(env, agent:AdvancedDQNAgent, episodes=1000,batch_size=64, render=False):
    """Train the agent on the environment"""
    scores = []
    win_rates = []
    balances = []
    total_time = time.time()
    print(f"=============== {np.random.rand()} =============")
    print(f"=============== {np.random.rand()} =============")
    # agent.load()
    for e in range(episodes):
        # Reset environment
        state = env.reset()
        done = False
        score = 0
        start_date = time.time()
        count = 0
        # print(f"========== {state['close']} ===========" )
        while not done:
            # Get action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Update score
            score += reward
            
            # Train agent
            if count % 10000 == 0:
                print(f"--> {count}")
                agent.replay()
                
            # Render if needed
            # if render and e % 10 == 0:
            #     env.render()
            # elapsed =time.time() - start_date
            # t = time.gmtime(elapsed)
            # print(time.strftime(f"-> date romaing for episode {e}: %H:%M:%S",t))
            
            # Update target model periodically
            if count % agent.update_target_frequency == 0:
                agent.update_target_model()
                
            count += 1
        agent.replay()

        
        # Save scores and metrics
        scores.append(score)
        win_rates.append(info['win_rate'] * 100)  # Convert to percentage
        balances.append(info['balance'])
        
        # Print progress
        # if e % 10 == 0:
        
        print(f"Episode: {e}/{episodes}, Score: {score:.2f}, Win Rate: {info['win_rate']*100:.2f}%, Balance: {info['balance']:.2f}, Epsilon: {agent.epsilon:.4f}")
        elapsed = time.time() - start_date
        t = time.gmtime(elapsed)
        print(time.strftime(f"date romaing for episode {e}: %H:%M:%S",t))
        # Save model periodically
        if e % 1 == 0:
            print(f"--> Model save in: dqn_trading_model/model_{version}_ep{e}.keras")
            agent.save(f"dqn_trading_model/model_{version}_ep{e}.keras")
    
    print(time.strftime(f"--> Total Time Training: %D:%H:%M:%S",time.gmtime(total_time)))
    # Save final model
    print(f"--> Model save in: dqn_trading_model/model_{version}_final.keras")
    # agent.save(f"dqn_trading_model/model_{version}_final.h5")
    agent.save()
    
    return scores, win_rates, balances

def backtest_model(env:EnhancedTradingEnv, agent, render=False):
    """Backtest the trained agent on the environment"""
    state = env.reset()
    done = False
    trades = []
    info = []
    test_metrics =[]
    count = 0
    while not done:
        # Get action (no exploration during testing)
        action = agent.act(state, training=False)
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Update state
        state = next_state
        test_metrics.append({
        'step': count,
        'price': info['price'],
        'action': action,
        'position': env.position,
        'balance': env.balance,
        'equity': info['equity'],
        'reward': reward,
        'market_trend': info['market_trend'],
        'volatility': info['volatility'],
        'win_rate': info['win_rate']
    })
        
        # Record trade if made
        if action != 0:  # If not hold
            trades.append((env.current_step, env.data.iloc[env.current_step]['close'], action))
        
        # Render if needed
        if render:
            env.render()
        count +=1
    
    win_rate = (env.winning_trades / env.total_trades) * \
        100 if env.total_trades > 0 else 0
    profit = env.balance - env.initial_balance

    test_results = pd.DataFrame(test_metrics)
    # Print backtest results
    print(f"Final Balance: {env.balance:.2f}")
    print(f"Total Profit: {profit:.2f} ({profit/ env.initial_balance*100:.2f}%)")
    print(f"Total Trades: {env.total_trades}")
    print(f"Winning Trades: {env.winning_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Max Drawdown: {min(test_results['equity']) - env.initial_balance:.2f}")
    
    # Print backtest results
    print(f"-->2  Final Balance: {info['balance']:.2f}")
    print(f"Total Trades: {info['trades']}")
    print(f"Win Rate: {info['win_rate']*100:.2f}%")
    print(f"Max Drawdown: {info['max_drawdown']*100:.2f}%")
    
    
    test_results.to_csv(f"backtest_results_{version}.csv", index=False)
    
    return info, trades

def plot_performance(scores, win_rates, balances, episodes):
    """Plot training performance metrics"""
    plt.figure(figsize=(15, 12))
    
    # Plot scores
    plt.subplot(3, 1, 1)
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    # Plot win rates
    plt.subplot(3, 1, 2)
    plt.plot(win_rates)
    plt.axhline(y=50, color='r', linestyle='--')
    plt.title('Win Rate (%)')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate (%)')
    
    # Plot balance
    plt.subplot(3, 1, 3)
    plt.plot(balances)
    plt.axhline(y=1000, color='r', linestyle='--')
    plt.title('Account Balance')
    plt.xlabel('Episode')
    plt.ylabel('Balance')
    
    plt.tight_layout()
    plt.savefig(f'dqn_trading_model/training_performance_{version}.png')
    plt.close()

def plot_trading_performance(data, trades, info):
    """Plot trading performance with buy/sell signals"""
    plt.figure(figsize=(15, 8))
    
    # Plot price
    plt.plot(data['close'], label='Close Price')
    
    # Plot buy signals
    buy_indices = [t[0] for t in trades if t[2] == 1]
    buy_prices = [t[1] for t in trades if t[2] == 1]
    plt.scatter(buy_indices, buy_prices, marker='^', color='g', label='Buy Signal', s=100)
    
    # Plot sell signals
    sell_indices = [t[0] for t in trades if t[2] == 2]
    sell_prices = [t[1] for t in trades if t[2] == 2]
    plt.scatter(sell_indices, sell_prices, marker='v', color='r', label='Sell Signal', s=100)
    
    # Add performance metrics as text
    plt.text(0.05, 0.95, f"Final Balance: ${info['balance']:.2f}\nWin Rate: {info['win_rate']*100:.2f}%\nTotal Trades: {info['trades']}\nMax Drawdown: {info['max_drawdown']*100:.2f}%", 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.title('Trading Performance with Buy/Sell Signals')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'dqn_trading_model/trading_performance_{version}.png')
    plt.close()
