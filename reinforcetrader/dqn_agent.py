import os

# Suppress TensorFlow logging for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import math
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, optimizers
import tensorflow.keras.ops as K
from tensorflow.keras.utils import plot_model

from typing import Any
from collections import deque
from tqdm import tqdm

from .state import EpisodeStateLoader

@keras.utils.register_keras_serializable()
# Define DQN Model Architecture
class DualBranchDQN(keras.Model):
    def __init__(self, motif_state_shape: tuple[int, int], context_state_size: int, action_size: int, learning_rate: float, dropout_p: float=0.1) -> None:
        super().__init__()

        # Motif Branch for finding 1-day price movement patterns using Conv1D
        motif_input = Input(shape=motif_state_shape, name='motif_input')
        motif_layer = layers.Conv1D(filters=32, kernel_size=1, padding='same', activation='relu')(motif_input)
        motif_layer = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(motif_layer)
        motif_layer = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(motif_layer)
        
        motif_max = layers.GlobalMaxPooling1D()(motif_layer)
        motif_avg = layers.GlobalAveragePooling1D()(motif_layer)
        
        motif_out = layers.Concatenate(name='motif_output')([motif_max, motif_avg])

        # Context Branch consisting of technical indicators
        context_input = Input(shape=(context_state_size,), name='context_input')
        context_layer = layers.Dense(units=256, activation='relu')(context_input)
        context_layer = layers.Dropout(dropout_p)(context_layer)
        context_layer = layers.Dense(units=128, activation='relu')(context_layer)
        context_layer = layers.Dropout(dropout_p)(context_layer)
        context_out = layers.Dense(units=64, activation='relu', name='context_output')(context_layer)
        
        # Late fusion of both the motif and context branches
        fused_branch = layers.Concatenate(name='late_fusion')([motif_out, context_out])
        fused_branch = layers.Dense(64, activation='relu')(fused_branch)
        fused_branch = layers.Dense(32, activation='relu')(fused_branch)
        
        # State and Advantage value stream layers for Dueling DQN
        V = layers.Dense(units=16, activation='relu')(fused_branch)
        V = layers.Dense(1, name='state_value')(V)
        A = layers.Dense(units=16, activation='relu')(fused_branch)
        A = layers.Dense(action_size, name='advantage_value')(A)
        
        # Combine to compute the Q values
        mA = K.mean(A, axis=1, keepdims=True)
        cA = layers.Subtract(name='center_advantage')([A, mA])
        Q = layers.Add(name='q_values')([V, cA])
        
        model_input = {'motif_input': motif_input, 'context_input': context_input}
        self._model = keras.Model(inputs=model_input, outputs=Q, name='DualBranchDQN')
        self._model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate, clipnorm=1.0))

    def get_model(self):
        return self._model

class DRLAgent:
    # Define constants for in-trade and out-of-trade positions
    OUT_TRADE = 0
    IN_TRADE = 1
    
    # Define constants that represent agent behaviour
    A_BUY = 0
    A_HOLD = 1
    A_SELL = 2
    
    def __init__(self, agent_config, reward_type: str='DSR', model_path: str | None=None) -> None:
        # Store state and action representation configurations
        self._window_size = agent_config['state_matrix_window']
        self._num_motif_feat = agent_config['num_motif_features']
        self._num_context_feat = agent_config['num_context_features']
        if reward_type not in {'DSR', 'DDDR', 'PNL'}:
            raise ValueError(f'Invalid reward type: {reward_type}')
        self.reward_type = reward_type
        
        # A = {0: buy, 1: hold, 2: sell}
        self._action_size = 3
        
        # Define masks to restrict action space based on trade position
        # A_out_trade = {0: buy, 1: hold}
        # A_in_trade  = {1: hold, 2: sell}
        self.in_trade_mask = tf.constant([-1e9, 0., 0.], dtype=tf.float32)
        self.out_trade_mask = tf.constant([0., 0., -1e9], dtype=tf.float32)
        self._get_mask = lambda pos: self.out_trade_mask if pos == DRLAgent.OUT_TRADE else self.in_trade_mask

        # Define memory to store (state, action, reward, new_state) pairs for exp. replay
        buffer_length = agent_config.get('memory_buffer_len', 10000)
        self._memory = deque(maxlen=buffer_length)
        
        # Check if the model path exists
        if not (model_path is None or os.path.exists(model_path)):
            raise FileNotFoundError(f'Model file at {model_path} does not exist')
        
        # Store the model name for pre loading a model
        self._model_path = model_path
        

        # Define parameters for behaviour policy and temporal learning
        self._gamma = agent_config.get('gamma', 0.95)  # discount factor
        self._epsilon = agent_config.get('epsilon_start', 1.0)
        self._epsilon_min = agent_config.get('epsilon_min', 0.01)
        self._epsilon_boost_factor = agent_config.get('epsilon_boost_factor', 0.0)
        
        if not 0.0 <= self._epsilon_boost_factor <= 1.0:
            raise ValueError('epsilon boost factor must be in [0.0, 1.0]')
        
        n_updates = agent_config.get('decay_updates', 25000) # No. of updates to minimum epsilon
        self._epsilon_decay = (self._epsilon_min / self._epsilon) ** (1.0 / n_updates)
        learning_rate = agent_config.get('learning_rate', 1e-3)
        dropout_p = agent_config.get('dropout_p', 0.1)


        # Load model or define new
        if self._model_path is not None:
            print(f'Loading model from {self._model_path}')
            self._model = keras.models.load_model(model_path)
        else:
            self._model = self._init_model(learning_rate, dropout_p)
        
        # Init target network update and frequency
        self._init_target_network()
    
    def get_model(self) -> keras.Model:
        return self._model

    def _init_model(self, learning_rate: float, dropout_p: float) -> keras.Model:
        # Compute state shapes for the model
        motif_shape = (self._window_size, self._num_motif_feat)
        
        # Compute the number of reward params to make its a valid MDP
        # Note: num_reward_params should have -1 (Rt is not included)
        # and +1 (trade pos is included). Thus net 0
        num_reward_params = len(self._reward_param_keys(self.reward_type))
        
        # Compute the context size
        context_size = self._window_size * self._num_context_feat + num_reward_params

        # Init model
        DDDQN = DualBranchDQN(motif_shape, context_size, self._action_size, learning_rate, dropout_p)

        return DDDQN.get_model()

    def _init_target_network(self) -> None:
        # Make a structural clone and copy weights
        self._target_model = keras.models.clone_model(self._model)
        self._target_model.set_weights(self._model.get_weights())


    def _update_target_network(self, tau: float = 1.0) -> None:
        # tau = 1.0 : hard update (copy weights exactly)
        # tau < 1.0 : soft/Polyak update (exponential moving average)
        
        online_weights = self._model.get_weights()
        target_weights = self._target_model.get_weights()

        new_weights = [
            tau * w_online + (1.0 - tau) * w_target
            for w_online, w_target in zip(online_weights, target_weights)]

        self._target_model.set_weights(new_weights)

    
    def _get_states(self, state_matrix: np.ndarray, extra_features: list[float]) -> dict[str, np.ndarray]:
        # Check the state matrix shape is correct
        total_features = self._num_motif_feat + self._num_context_feat
        if state_matrix.shape != (self._window_size, total_features):
            raise ValueError(f'Invalid state matrix shape: {state_matrix.shape}')
        
        # Define the motif input
        # Expand dims from [window, num_features] to [batch_size=1, window, num_features]
        motif_input = np.expand_dims(state_matrix[:, :self._num_motif_feat], axis=0).astype(np.float32)

        # Flatten the context input and combine with other features (e.g. trade pos, reward params)
        context_flat = state_matrix[:, self._num_motif_feat:].astype(np.float32).reshape(1, -1)
        extra_context = np.array([extra_features], dtype=np.float32)
        context_input = np.concatenate([context_flat, extra_context], axis=1)

        return {'motif_input': motif_input, 'context_input': context_input}
    
    def _get_q_values(self, state_matrix: np.ndarray, extra_features: list[float]) -> tf.Tensor:
        # Predict q values through DQN        
        model_input = self._get_states(state_matrix, extra_features)
        q_values = self._model(model_input, training=False)
        
        return q_values
    
    def _act(self, state_matrix: np.ndarray, trade_pos: int, extra_features: list[float], training: bool=True) -> int:
        if trade_pos not in {DRLAgent.IN_TRADE, DRLAgent.OUT_TRADE}:
            raise ValueError(f'Invalid trade position: {trade_pos}')
        
        # Defines an epsilon-greedy behaviour policy
        # Pick random action epsilon times
        if training and np.random.random() < self._epsilon:
            if trade_pos == DRLAgent.OUT_TRADE:
                # A_{Out of trade}: {0: buy, 1: hold (out)}
                return np.random.randint(0, 1)
            else:
                # A_{In trade}: {1: hold (in), 2: sell}
                return np.random.randint(1, 2)

        # Pick action from DQN with probability 1 - epsilon
        # Dont use dropout as act() function is not used for gradient descent updates
        
        # Construct the extra set of computes used to construct state representation
        extra_context = [trade_pos] + extra_features
        
        # Compute Q values from DQN and restrict action space
        q_values = self._get_q_values(state_matrix, extra_context) + self._get_mask(trade_pos) # type: ignore
        action = int(tf.argmax(q_values[0], axis=-1, output_type=tf.int32).numpy())
        
        return action

    def _exp_replay(self, batch_size: int) -> float:
        if len(self._memory) < batch_size:
            raise ValueError('Not enough samples in memory to perform experience replay')

        # Prefer uniform sampling to break time correlations
        idx = np.random.choice(len(self._memory), size=batch_size, replace=False)
        batch = [self._memory[i] for i in idx]

        # Prepare the batch arrays
        B, W, M, C = batch_size, self._window_size, self._num_motif_feat, self._num_context_feat
        extra_context_len = len(self._reward_param_keys(self.reward_type)) # (...) + 1 - 1
        motif_batch = np.empty((B, W, M), dtype=np.float32)
        context_batch = np.empty((B, W*C + extra_context_len), dtype=np.float32)
        next_motif_batch = np.empty((B, W, M), dtype=np.float32)
        next_context_batch = np.empty((B, W*C + extra_context_len), dtype=np.float32)
        actions = np.empty((B,), dtype=np.int32)
        rewards = np.empty((B,), dtype=np.float32)
        
        # Create a mask to restrict the action space
        next_action_mask = []

        for i, (state, curr_ef, prev_pos, action, reward, next_state, next_ef, curr_pos) in enumerate(batch):
            # Extract state components for feeding to DQN braches
            # Compute the current state features for dual branch
            motif = state[:, :M].astype(np.float32)
            ctx = state[:, M:].astype(np.float32).reshape(-1)
            motif_batch[i] = motif
            context_batch[i] = np.concatenate([ctx, [prev_pos], curr_ef], axis=0, dtype=np.float32)
            
            # Compute the next state features for dual branch
            next_motif = next_state[:, :M].astype(np.float32)
            next_ctx = next_state[:, M:].astype(np.float32).reshape(-1)  # [W*C]
            next_motif_batch[i] = next_motif
            next_context_batch[i] = np.concatenate([next_ctx, [curr_pos], next_ef], axis=0, dtype=np.float32)
            
            # Add the appropriate mask based on curr_pos
            # NOTE: Action mask is used for masking next q values.
            # Thus, curr_pos is used instead of prev_pos
            next_action_mask.append(self._get_mask(curr_pos))
            
            # Set the action and reward
            actions[i] = action
            rewards[i] = reward
        
        # Create tensor for action mask and rewards for arithmetic later
        next_action_mask = tf.stack(next_action_mask, axis=0)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        
        # Prepare current and next state inputs
        curr_state = {'motif_input': motif_batch, 'context_input': context_batch}
        next_state = {'motif_input': next_motif_batch, 'context_input': next_context_batch}
        
        # Predict Q values for current and next states and restrict action space
        # NOTE: training=False for not using dropout layers.
        q_current = self._model(curr_state, training=False)
        q_next_online = self._model(next_state, training=False) + next_action_mask
        q_next_target = self._target_model(next_state, training=False) + next_action_mask
        
        # Double DQN next q computation
        # Note: use online table to get action with highest q value for next state
        # Then, use the target table q table to compute q value for next state
        a_star = tf.argmax(q_next_online, axis=1, output_type=tf.int32)
        batch_indices = tf.range(B, dtype=tf.int32)
        max_q_next = tf.gather_nd(q_next_target, tf.stack([batch_indices, a_star], axis=1))
        
        # Compute observed returns using Bellman equation
        returns = rewards + (self._gamma * max_q_next)
        
        # Create targets by updating only the taken actions' q values
        target_actions = tf.stack([batch_indices, tf.convert_to_tensor(actions, dtype=tf.int32)], axis=1)
        targets = tf.tensor_scatter_nd_update(q_current, target_actions, returns)

        loss = self._model.train_on_batch(curr_state, targets)
        self._update_target_network(tau=0.005)

        return float(loss)

    def _reward_param_keys(self, reward_type: str) -> set[str]:
        # Small Helper method. Uff..
        match reward_type:
            case 'DSR':
                return {'Rt', 'A_tm1', 'B_tm1'}
            case 'DDDR':
                return {'Rt', 'A_tm1', 'DD_tm1'}
            case 'PNL':
                return {'Rt'}
            case _:
                raise ValueError(f'Invalid reward type: {reward_type}')
    
    def _compute_reward(self, params: dict[str, float], action: float) -> float:
        # Check if all the required keys are present        
        missing_params = [k for k in self._reward_param_keys(self.reward_type) if k not in params]
        if missing_params:
            raise ValueError(f"Missing required reward parameters for {self.reward_type}: {missing_params}")
        
        # Determine if transaction costs are applicable
        if action == DRLAgent.A_HOLD:
            tc = 0 # No cost for holding
        else:
            tc = 0.25 / 100 # 0.25%
        
        # Compute the reward based on the selected type
        match self.reward_type:
            case 'DSR':
                return self._DSR(params['Rt'], params['A_tm1'], params['B_tm1'], tc)
            case 'DDDR':
                return self._DDDR(params['Rt'], params['A_tm1'], params['DD_tm1'], tc)
            case 'PNL':
                return self._PnL(params['Rt'], tc)
            case _:
                raise ValueError(f'Invalid reward type: {self.reward_type}')
            
    
    def _DSR(self, Rt, A_tm1, B_tm1, tc, eps: float = np.finfo(float).eps) -> float:
        # Discount the returns by the transaction cost
        Rtd = Rt - tc
        
        # Compute delta values
        dAt = Rtd - A_tm1
        dBt = Rtd ** 2 - B_tm1

        num = B_tm1 * dAt - 0.5 * A_tm1 * dBt
        denom = (B_tm1 - A_tm1 ** 2) ** 1.5 + eps

        return num / denom
    
    def _PnL(self, Rt, tc: float) -> float:
        # Discount the returns by the transaction cost
        Rtd = Rt - tc
        
        # Simple profit and loss reward
        return np.log1p(Rtd)

    
    def _DDDR(self, Rt, A_tm1, DD_tm1, tc, eps: float=np.finfo(float).eps) -> float:
        # Compute the Differential Downside Deviation Ratio
        # (as derived in https://doi.org/10.1109/72.935097)
        
        # Discount the returns by the transaction cost
        Rtd = Rt - tc
        
        if Rtd > 0:
            num = Rtd - 0.5 * A_tm1
            denom = DD_tm1 + eps
        else:
            num = (DD_tm1 ** 2) * (Rtd - 0.5 * A_tm1) - (0.5 * A_tm1 * Rtd ** 2)
            denom = DD_tm1 ** 3 + eps
                
        return num / denom
    
    def _init_reward_state_computes(self):
        # Helper method to init/reset the compute values
        self._r_A_tm1 = 0
        self._r_B_tm1 = 0
        self._r_DD2_tm1 = 0
    
    def _get_reward_state_computes(self) -> dict[str, float]:
        match self.reward_type:
            case 'DSR':
                return {'A_tm1': self._r_A_tm1, 'B_tm1': self._r_B_tm1}
            case 'DDDR':
                return {'A_tm1': self._r_A_tm1, 'DD_tm1': self._r_DD2_tm1 ** 0.5}
            case _:
                return {}
    
    def _update_reward_computes(self, Rt, n: float=1e-4) -> dict[str, float]:
        match self.reward_type:
            case 'DSR':
                self._r_A_tm1 += n * (Rt - self._r_A_tm1)
                self._r_B_tm1 += n * (Rt ** 2 - self._r_B_tm1)
            case 'DDDR':
                self._r_A_tm1 += n * (Rt - self._r_A_tm1)
                self._r_DD2_tm1 += n * (min(Rt, 0) ** 2 - self._r_DD2_tm1)
            case 'PNL':
                pass
        return self._get_reward_state_computes()

    def train(self, state_loader: EpisodeStateLoader, episode_ids: list[int], train_config: dict[str, Any], reward_diag: bool=False) -> dict[str, Any]:
        # Make req. directories if not exist
        os.makedirs(train_config['model_dir'], exist_ok=True)
        os.makedirs(train_config['plots_dir'], exist_ok=True)
        os.makedirs(train_config['logs_dir'], exist_ok=True)

        # Extract training parameters from config
        replay_start_size = train_config.get('replay_start_size', 5000)
        train_interval = train_config.get('train_interval', 1)
        batch_size = train_config.get('batch_size', 256)
        
        logs_by_episode = {}

        all_tickers = state_loader.get_all_tickers()

        for e in episode_ids:

            # Compute the total loss across the whole episode
            train_loss = 0.0
            # Track env steps
            env_steps = 0
            # store the current value of epsilon for boosting
            epsilon_start = self._epsilon
                    
            # Define parameters for episode-ticker iteration
            L = state_loader.get_episode_len('train', e)
            t0 = self._window_size - 1
            
            # Initialize container to store received rewards
            if reward_diag:
                episode_rewards = np.zeros((len(all_tickers), L - t0 - 1))
            
            # Iterate tickers, training sequentially
            for ti, ticker in enumerate(tqdm(all_tickers, desc=f'Training episode {e}', ncols=100)):

                state = state_loader.get_state_matrix('train', e, ticker, t0, self._window_size)
                prev_pos = DRLAgent.OUT_TRADE
                
                # Initalize the reward computes for the ticker
                self._init_reward_state_computes()
                    
                for t in range(t0, L - 1):
                    # Get the reward computes that are included in the state representation
                    reward_computes = self._get_reward_state_computes()
                    # Store the current extra features (ef) used in state rep.
                    curr_ef = list(reward_computes.values())
                    
                    # Derive action based on eps-greedy policy
                    # NOTE: Here, extra features are some computes that are used
                    # to compute the reward. To make a valid MDP, there variables
                    # are included in the state representation
                    action = self._act(state, prev_pos, curr_ef)
                    
                    # Compute the current trade position based on new action
                    if prev_pos == DRLAgent.OUT_TRADE and action == DRLAgent.A_BUY:
                        curr_pos = DRLAgent.IN_TRADE
                    elif prev_pos == DRLAgent.IN_TRADE and action == DRLAgent.A_SELL:
                        curr_pos = DRLAgent.OUT_TRADE
                    else:
                        curr_pos = prev_pos
                    
                    # Append the return value for reward calculation
                    if curr_pos == DRLAgent.IN_TRADE:
                        Rt = state_loader.get_reward_computes('train', e, ticker, t)['1DFRet']
                    else:
                        Rt = 0
                        
                    reward_computes['Rt'] = Rt
                    
                    # Compute and store the reward value for the state-action pair
                    reward = self._compute_reward(reward_computes, action)
                    if reward_diag:
                        episode_rewards[ti, t - t0] = reward

                    # Get the next state
                    next_state = state_loader.get_state_matrix('train', e, ticker, t + 1, self._window_size)

                    # Update the reward compute variables and store the computes features
                    next_ef = list(self._update_reward_computes(Rt).values())
                    
                    # store transition
                    self._memory.append((state, curr_ef, prev_pos, action, reward, next_state, next_ef, curr_pos))

                    # Update env steps taken
                    env_steps += 1
                    
                    # train from replay if enough samples; accumulate training loss for this group
                    if len(self._memory) >= replay_start_size:
                        # Train every train_interval steps
                        if env_steps % train_interval == 0:
                            loss = self._exp_replay(batch_size)
                            train_loss += loss
                                
                            # Decay epsilon slowly until min
                            if self._epsilon > self._epsilon_min:
                                self._epsilon *= self._epsilon_decay
                    
                    # Advance to the next state
                    state = next_state
                    prev_pos = curr_pos

            # Plot reward diagostics
            if reward_diag:
                self._visualize_rewards(episode_rewards, f'Epsiode {e} Reward Visualization')
            
            # Run validation on this episode's validation set
            val_result = self._run_validation(state_loader, e, all_tickers)
            
            # Print the validation summary
            print(f'Episode {e} validation summary:')
            print(f"Train loss: {train_loss:.4f}, Cum. Reward: {val_result['cum_reward']:.4f}, Total val trades: {val_result['total_trades']}, Hit rate: {val_result['hit_rate']:.2f}")
            print(f"Trade Duration: {val_result['trade_duration']:.2f}, Total PnL: {val_result['total_pnl']:.2f}, Profit Factor: {val_result['profit_factor']:.3f}")
            print(f"Force End Trade Count: {val_result['force_end_trades']}, Force End PnL: {val_result['force_end_pnl']:.2f}")
            
            # Boost the epsilon a bit for next episode (as every episode has diff regimes)
            epsilon_end = self._epsilon
            self._epsilon = epsilon_end + self._epsilon_boost_factor * (epsilon_start - epsilon_end)
            
            # Store logs for this episode
            logs_by_episode[e] = {
                'train_loss': train_loss,
                'val_results': val_result,
                'epsilon_start': epsilon_start,
                'epsilon_current': self._epsilon,
                'epsilon_end': epsilon_end
            }
            
        # Plot all the training and validation losses
        train_losses = [logs_by_episode[ep]['train_loss'] for ep in episode_ids]
        val_losses = [-logs_by_episode[ep]['val_results']['cum_reward'] for ep in episode_ids]
        self._plot_losses(train_losses, val_losses, fname=os.path.join(train_config['plots_dir'], 'train_losses.png'))
        
        # Also plot the epsilon decay
        eps_start = [logs_by_episode[ep]['epsilon_start'] for ep in episode_ids] 
        eps_curr = [logs_by_episode[ep]['epsilon_current'] for ep in episode_ids] 
        eps_end = [logs_by_episode[ep]['epsilon_end'] for ep in episode_ids] 
        eps_fname = os.path.join(train_config['plots_dir'], 'epsilon_decay.png')
        self._plot_epsilon_decay(eps_start, eps_curr, eps_end, fname=eps_fname)
        
        # Save model checkpoint with the current date and time
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._model.save(os.path.join(train_config['model_dir'], f'model_{date_str}.keras'))
        
        # Save logs to a json file
        with open(os.path.join(train_config['logs_dir'], f'train_logs_{date_str}.json'), 'w') as f:
            json.dump(logs_by_episode, f, indent=2)

        return logs_by_episode

    def _visualize_rewards(self, reward_data, plot_name: str):
        # Create plot with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})
        
        # Left plot: Cumulative sum of reward (i.e, approximations of e.g., Sharpe ratio)
        
        num_tickers, num_steps = reward_data.shape
        for ti in range(num_tickers):
            ax1.plot(np.arange(num_steps), reward_data[ti, :])
        ax1.set_title('Cumulative Sum of Rewards Per Ticker')
        ax1.set_ylabel('Reward')
        
        # Right Plot: Distribution of the receieved rewards
        reward_flat = reward_data.ravel()
        ax2.hist(reward_flat, edgecolor='black', label='Histogram')
        ax2.axvline(x=np.mean(reward_flat), linestyle='--', label='Mean')
        ax2.axvline(x=np.median(reward_flat), linestyle=':', label='Median')
        ax2.legend()
        ax2.set_title('Reward Distribution')
        ax2.set_ylabel('Frequency')
        
        plt.show()
        
        
        
    
    def _run_validation(self, state_loader: EpisodeStateLoader, episode_id: int, tickers: list[str]) -> dict[str, int | float]:
        # The agent does no exploration, only exploitation
        # Define metrics to track for validation cycle
        cum_reward = 0.0
        total_trades = 0
        winning_trade_count = 0
        losing_trade_count = 0
        gross_profit = 0.0
        gross_loss = 0.0
        in_trade_days = 0
        
        # Track trades that had to be forcefully closed as episode ends
        # Tells us the distorition in performance metrics
        force_end_trades = 0
        force_end_pnl = 0.0
        
        L = state_loader.get_episode_len('validate', episode_id)
        
        for ticker in tickers:
            t0 = self._window_size - 1
            state = state_loader.get_state_matrix('validate', episode_id, ticker, t0, self._window_size)
            prev_pos = DRLAgent.OUT_TRADE
            entry_price = None

            # Initalize the reward computes for the ticker
            self._init_reward_state_computes()
            
            for t in range(t0, L - 1):
                # Get the reward computes that are included in the state representation
                reward_computes = self._get_reward_state_computes()
                # Store the current extra features (ef) for deciding action
                curr_ef = list(reward_computes.values())
                
                # training=False turns off exploration
                action = self._act(state, prev_pos, curr_ef, training=False)
                curr_price = state_loader.get_state_OHLCV('validate', episode_id, ticker, t)['Close']
                
                # Compute the current trade position based on new action
                if prev_pos == DRLAgent.OUT_TRADE and action == DRLAgent.A_BUY:
                    curr_pos = DRLAgent.IN_TRADE
                elif prev_pos == DRLAgent.IN_TRADE and action == DRLAgent.A_SELL:
                    curr_pos = DRLAgent.OUT_TRADE
                else:
                    curr_pos = prev_pos
                
                # Append the return value for reward calculation
                if curr_pos == DRLAgent.IN_TRADE:
                    Rt = state_loader.get_reward_computes('validate', episode_id, ticker, t)['1DFRet']
                else:
                    Rt = 0
                
                # Append Rt for reward calculation (used by e.g. DSR)    
                reward_computes['Rt'] = Rt
                
                # Compute the reward value for the state-action pair
                reward = self._compute_reward(reward_computes, action)
                cum_reward += reward

                # Trade performance tracking
                if curr_pos == DRLAgent.IN_TRADE:
                    # Track how many days the agent is in trade
                    in_trade_days += 1

                if prev_pos == DRLAgent.OUT_TRADE and curr_pos == DRLAgent.IN_TRADE:
                    entry_price = curr_price
                    total_trades += 1
                elif prev_pos == DRLAgent.IN_TRADE and (curr_pos == DRLAgent.OUT_TRADE or t == L - 2):
                    if entry_price is None:
                        raise ValueError('Entry price unknown on trade close')
                    
                    # Compute the trade pnl
                    trade_pnl = curr_price - entry_price
                    
                    # Count how many trades had to be forcefully closed at the end
                    if t == L - 2:
                        force_end_trades += 1
                        force_end_pnl += trade_pnl
                    
                    if trade_pnl >= 0:
                        gross_profit += trade_pnl
                        winning_trade_count += 1
                    else:
                        gross_loss += trade_pnl
                        losing_trade_count += 1
                        
                    # Reset entry price for future trades
                    entry_price = None

                next_state = state_loader.get_state_matrix('validate', episode_id, ticker, t + 1, self._window_size)
                
                # Update the reward computes for the next iteration
                self._update_reward_computes(Rt)
                
                # Advance to the next state
                state =  next_state
                prev_pos = curr_pos

        # WARNING: Metrics consider forcefully ended trades which could skew performance
        total_pnl = gross_profit + gross_loss
        hit_rate = winning_trade_count / max(total_trades, 1)
        trade_duration = in_trade_days / max(total_trades, 1)
        profit_factor = gross_profit / max(abs(gross_loss), 1e-12)
        
        return {
            'cum_reward': cum_reward,
            'total_trades': total_trades,
            'trade_duration': trade_duration,
            'hit_rate': hit_rate,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'force_end_trades': force_end_trades,
            'force_end_pnl': force_end_pnl
        }

    def _plot_epsilon_decay(self, eps_start, eps_curr, eps_end: list[float], fname: str | None = None, show: bool = True):
        x = np.arange(1, len(eps_curr) + 1)

        plt.figure(figsize=(8, 4))
        plt.plot(x, eps_start, marker=9, linestyle='--', color='tab:gray', label='Epsilon Start')
        plt.plot(x, eps_curr, marker='o', linewidth=2, color='tab:green', label='Current Epsilon')
        plt.plot(x, eps_end, marker=8, linestyle='--', color='tab:gray', label='Epsilon End')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay over Episodes')
        plt.grid(True, linestyle='-', alpha=0.2)

        if fname:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            plt.savefig(fname, dpi=150)

        if show:
            plt.show()
        else:
            plt.close()
    
    def _plot_losses(self, train_losses: list[float], val_losses: list[float], fname: str | None = None, show: bool = True):

        x = np.arange(1, len(train_losses) + 1)

        fig, ax1 = plt.subplots(figsize=(10, 4))

        # Left axis: training loss
        ax1.plot(x, train_losses, marker='o', linewidth=2, color='tab:blue',
                label='Train loss (MSE per episode)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Train loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, linestyle='--', alpha=0.3)

        # Right axis: validation loss
        ax2 = ax1.twinx()
        ax2.plot(x, val_losses, marker='s', linewidth=2, color='tab:orange',
                label='Validation loss (−sum reward per episode)')
        ax2.set_ylabel('Validation loss', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Combine legends
        lines, labels = [], []
        for ax in [ax1, ax2]:
            l, lab = ax.get_legend_handles_labels()
            lines.extend(l)
            labels.extend(lab)
        ax1.legend(lines, labels, loc='best')

        plt.title('Training Vs. Validation Loss per Episode')
        fig.tight_layout()

        if fname:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            plt.savefig(fname, dpi=150)

        if show:
            plt.show()
        else:
            plt.close()

    def _action_to_onehot(self, a: int) -> dict:
        # A = {0: buy, 1: hold, 2: sell}
        return {
            'buy':  1 if a == DRLAgent.A_BUY else 0,
            'hold':  1 if a == DRLAgent.A_HOLD else 0,
            'sell': 1 if a == DRLAgent.A_SELL else 0,
        }

    def test(self, state_loader: EpisodeStateLoader, episode_id: int, test_config: dict[str, Any]):
        # NOTE: Assumes no exploration and only exploitation
        all_tickers = state_loader.get_all_tickers()

        # Keep a single ordered list of tickers and parallel lists of series for safe alignment
        col_keys = [] # ['AAPL', 'MSFT', ...] in deterministic order
        sig_series_list = [] # [Series-of-dicts, ...]
        px_series_list  = [] # [Series-of-floats, ...]
        
        L = state_loader.get_episode_len('test', episode_id)
        
        # Get the index for the signals and prices dataframes
        df_idx = state_loader.get_test_dates(episode_id)
        
        for ticker in tqdm(all_tickers, desc=f'Testing episode {episode_id}', ncols=100):
            # Prepare for the first iteration
            # NOTE: t0 can be init to 0 instead of window size - 1 because
            # of pad_overflow functionality in get_state_matrix
            t0 = 0
            state = state_loader.get_state_matrix('test', episode_id, ticker, t0, self._window_size)
            prev_pos = DRLAgent.OUT_TRADE

            # Allocate containers (length L to match idx)
            sig_cells = [None] * L
            close_px  = np.empty(L, dtype=np.float32)

            # main test loop
            for t in range(t0, L - 1):
                # Get the reward computes that are included in the state representation
                reward_computes = self._get_reward_state_computes()
                # Store the current extra features (ef) for deciding action
                curr_ef = list(reward_computes.values())
                
                # Get the current close price
                curr_close = state_loader.get_state_OHLCV('test', episode_id, ticker, t)['Close']
                close_px[t] = curr_close

                # Derive action based on greedy policy
                action = self._act(state, prev_pos, curr_ef, training=False)
                sig_cells[t] = self._action_to_onehot(action) # type: ignore

                # Compute the current trade position based on new action
                if prev_pos == DRLAgent.OUT_TRADE and action == DRLAgent.A_BUY:
                    curr_pos = DRLAgent.IN_TRADE
                elif prev_pos == DRLAgent.IN_TRADE and action == DRLAgent.A_SELL:
                    curr_pos = DRLAgent.OUT_TRADE
                else:
                    curr_pos = prev_pos
                
                # Advance to the next state and update prev_pos
                next_state = state_loader.get_state_matrix('test', episode_id, ticker, t + 1, self._window_size)
                state = next_state
                prev_pos = curr_pos

            # Final row (t = L-1): record price; no new decision possible so force hold-out or sell
            close_px[L - 1] = state_loader.get_state_OHLCV('test', episode_id, ticker, L - 1)['Close']
            if sig_cells[L - 1] is None:
                if prev_pos == DRLAgent.OUT_TRADE:
                    # Hold from taking a trade
                    sig_cells[L - 1] = self._action_to_onehot(DRLAgent.A_HOLD) # type: ignore
                else:
                    # Close the trade
                    sig_cells[L - 1] = self._action_to_onehot(DRLAgent.A_SELL) # type: ignore

            # Stash in ordered lists
            col_keys.append(ticker)
            sig_series_list.append(pd.Series(sig_cells, index=df_idx))
            px_series_list.append(pd.Series(close_px, index=df_idx))

        # Concatenate into DataFrames with single-level 'Ticker' columns
        if len(sig_series_list) == 0:
            signals_df = pd.DataFrame()
            prices_df  = pd.DataFrame()
        else:
            # Use outer join for safety in case of rare index mismatches
            signals_df = pd.concat(sig_series_list, axis=1, join='outer')
            prices_df  = pd.concat(px_series_list,  axis=1, join='outer')

            # Set columns to a simple Index of tickers
            signals_df.columns = pd.Index(col_keys, name='Ticker')
            prices_df.columns  = pd.Index(col_keys, name='Ticker')

        # Save artifacts
        out_dir = test_config.get('outputs_dir')
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            signals_path = os.path.join(out_dir, f'signals_{ts}.pkl')
            prices_path  = os.path.join(out_dir, f'prices_{ts}.pkl')
            signals_df.to_pickle(signals_path)
            prices_df.to_pickle(prices_path)
            print(f'Saved signals to {signals_path}')
            print(f'Saved prices  to {prices_path}')

        return signals_df, prices_df