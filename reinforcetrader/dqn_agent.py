import os

# Suppress TensorFlow logging for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings

import json
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from scipy import stats
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
    def __init__(self, state_shape: tuple[int, int], num_reward_feat: int, action_size: int, learning_rate: float, dropout_p: float=0.1) -> None:
        super().__init__()

        # All state features go through Conv1D
        state_input = Input(shape=state_shape, name='state_input')
        state_layer = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(state_input)
        state_layer = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(state_layer)
        state_layer = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(state_layer)
        
        state_max = layers.GlobalMaxPooling1D()(state_layer)
        state_avg = layers.GlobalAveragePooling1D()(state_layer)
        
        state_out = layers.Concatenate(name='state_output')([state_max, state_avg])

        # Second branch for reward features (e.g., trade_pos, A_t, B_t, DD_t, etc.)
        reward_input = Input(shape=(num_reward_feat,), name='reward_input')
        reward_layer = layers.Dense(units=32, activation='relu')(reward_input)
        reward_layer = layers.Dropout(dropout_p)(reward_layer)
        reward_layer = layers.Dense(units=16, activation='relu')(reward_layer)
        reward_layer = layers.Dropout(dropout_p)(reward_layer)
        reward_out = layers.Dense(units=4, activation='relu', name='reward_output')(reward_layer)
        
        # Late fusion of both the state and reward branches
        fused_branch = layers.Concatenate(name='late_fusion')([state_out, reward_out])
        fused_branch = layers.Dense(64, activation='relu')(fused_branch)
        fused_branch = layers.Dense(32, activation='relu')(fused_branch)
        
        # State and Advantage value stream layers for Dueling DQN
        V = layers.Dense(units=16, activation='relu')(fused_branch)
        V = layers.Dense(1, name='state_value')(V)
        A = layers.Dense(units=16, activation='relu')(fused_branch)
        A = layers.Dense(action_size, name='advantage_value')(A)
        
        # Combine to compute the Q values
        # Q = V(s) + (A(s,a) - mean(A(s,*)))
        mA = K.mean(A, axis=1, keepdims=True)
        cA = layers.Subtract(name='center_advantage')([A, mA])
        
        Q = layers.Add(name='q_values')([V, cA])
        
        model_input = {'state_input': state_input, 'reward_input': reward_input}
        self._model = keras.Model(inputs=model_input, outputs=Q, name='DualBranchDQN')
        self._model.compile(loss=keras.losses.Huber(), optimizer=optimizers.Adam(learning_rate, clipnorm=1.0))

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
        self._num_features = agent_config['num_features']
        if reward_type not in {'DSR', 'DDDR', 'PNL'}:
            raise ValueError(f'Invalid reward type: {reward_type}')
        self.reward_type = reward_type
        
        # Compute the number of reward params to make its a valid MDP
        # Note: num_reward_params should have -1 (Rt is not included)
        # and +1 (trade pos is included). Thus net 0
        self.num_reward_params = len(self._reward_param_keys(self.reward_type))
        
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

        # Extract training parameters from config
        self.replay_start_size = agent_config.get('replay_start_size', 5000)
        self.train_interval = agent_config.get('train_interval', 1)
        self.batch_size = agent_config.get('batch_size', 256)

        # Load model or define new
        if self._model_path is not None:
            print(f'Loading model from {self._model_path}')
            self._model = keras.models.load_model(self._model_path)
        else:
            self._model = self._init_model(learning_rate, dropout_p)
        
        # Init target network update and frequency
        self._init_target_network()
        
        # Init exp reply batch arrays
        self.init_exp_replay_batches()
    
    def get_model(self) -> keras.Model:
        return self._model

    def _init_model(self, learning_rate: float, dropout_p: float) -> keras.Model:
        # Compute state shapes for the model
        state_shape = (self._window_size, self._num_features)

        # Init the DQN model
        DDDQN = DualBranchDQN(state_shape, self.num_reward_params, self._action_size, learning_rate, dropout_p)

        return DDDQN.get_model()

    def plot_model_arch(self, fname: str | None=None) -> None:
        if fname is not None and fname:
            return plot_model(self._model, to_file=fname, show_shapes=True, show_layer_names=True)
        
        return plot_model(self._model, show_shapes=True, show_layer_names=True)
    
    def _init_target_network(self) -> None:
        # Make a structural clone and copy weights
        self._target_model = keras.models.clone_model(self._model)
        self._target_model.set_weights(self._model.get_weights())

    def init_exp_replay_batches(self) -> None:
        # Create batch arrays for experience replay
        B, W, F, R = self.batch_size, self._window_size, self._num_features, self.num_reward_params
        self.state_batch = np.empty((B, W, F), dtype=np.float32)
        self.rwd_params_batch = np.empty((B, R), dtype=np.float32)
        self.next_state_batch = np.empty((B, W, F), dtype=np.float32)
        self.next_rwd_params_batch = np.empty((B, R), dtype=np.float32)
        self.actions = np.empty((B,), dtype=np.int32)
        self.rewards = np.empty((B,), dtype=np.float32)

    def _update_target_network(self, tau: float = 1.0) -> None:
        # tau = 1.0 : hard update (copy weights exactly)
        # tau < 1.0 : soft/Polyak update (exponential moving average)
        
        online_weights = self._model.get_weights()
        target_weights = self._target_model.get_weights()

        new_weights = [
            tau * w_online + (1.0 - tau) * w_target
            for w_online, w_target in zip(online_weights, target_weights)]

        self._target_model.set_weights(new_weights)

    
    def _get_states(self, state_matrix: np.ndarray, reward_params: list[float]) -> dict[str, np.ndarray]:
        # Check the state matrix shape is correct
        if state_matrix.shape != (self._window_size, self._num_features):
            raise ValueError(f'Invalid state matrix shape: {state_matrix.shape}')
        
        # Define the state input
        # Expand dims from [window, num_features] to [batch_size=1, window, num_features]
        state_input = np.expand_dims(state_matrix, axis=0).astype(np.float32)

        # Create a numpy array for reward params
        reward_input = np.array([reward_params], dtype=np.float32)

        return {'state_input': state_input, 'reward_input': reward_input}
    
    def _get_q_values(self, state_matrix: np.ndarray, reward_params: list[float]) -> tf.Tensor:
        # Predict q values through DQN        
        model_input = self._get_states(state_matrix, reward_params)
        q_values = self._model(model_input, training=False)
        
        return q_values
    
    def _act(self, state_matrix: np.ndarray, trade_pos: int, reward_params: list[float], training: bool=True) -> tuple[int, float | None]:
        if trade_pos not in {DRLAgent.IN_TRADE, DRLAgent.OUT_TRADE}:
            raise ValueError(f'Invalid trade position: {trade_pos}')
        
        # Defines an epsilon-greedy behaviour policy
        # Pick random action epsilon times
        if training and np.random.random() < self._epsilon:
            if trade_pos == DRLAgent.OUT_TRADE:
                # A_{Out of trade}: {0: buy, 1: hold (out)}
                return (np.random.randint(0, 2), None)
            else:
                # A_{In trade}: {1: hold (in), 2: sell}
                return (np.random.randint(1, 3), None)

        # Pick action from DQN with probability 1 - epsilon
        # Dont use dropout as act() function is not used for gradient descent updates
        
        # Construct the extra set of computes used to construct state representation
        all_reward_params = [trade_pos] + reward_params
        
        # Compute Q values from DQN and restrict action space
        q_values = self._get_q_values(state_matrix, all_reward_params) + self._get_mask(trade_pos) # type: ignore
        action = int(tf.argmax(q_values[0], axis=-1, output_type=tf.int32).numpy())
        
        pred_q_value = float(q_values[0, action])
        return (action, pred_q_value)

    def _exp_replay(self) -> float:
        if len(self._memory) < self.batch_size:
            raise ValueError('Not enough samples in memory to perform experience replay')

        # Prefer uniform sampling to break time correlations
        idx = np.random.choice(len(self._memory), size=self.batch_size, replace=False)
        batch = [self._memory[i] for i in idx]
        
        # Create a mask to restrict the action space
        next_action_mask = []

        for i, (state, curr_ef, prev_pos, action, reward, next_state, next_ef, curr_pos) in enumerate(batch):
            # Compute the current state features for dual branch
            self.state_batch[i] = state
            self.rwd_params_batch[i] = np.concatenate([[prev_pos], curr_ef], axis=0, dtype=np.float32)
            
            # Compute the next state features for dual branch
            self.next_state_batch[i] = next_state
            self.next_rwd_params_batch[i] = np.concatenate([[curr_pos], next_ef], axis=0, dtype=np.float32)
            
            # Add the appropriate mask based on curr_pos
            # NOTE: Action mask is used for masking next q values.
            # Thus, curr_pos is used instead of prev_pos
            next_action_mask.append(self._get_mask(curr_pos))
            
            # Set the action and reward
            self.actions[i] = action
            self.rewards[i] = reward
        
        # Create tensor for action mask and rewards for arithmetic later
        next_action_mask = tf.stack(next_action_mask, axis=0)
        # NOTE: Not self.rewards for code breaking due to looping logic above
        rewards = tf.convert_to_tensor(self.rewards, dtype=tf.float32)
        
        # Prepare current and next state inputs
        curr_state = {'state_input': self.state_batch, 'reward_input': self.rwd_params_batch}
        next_state = {'state_input': self.next_state_batch, 'reward_input': self.next_rwd_params_batch}
        
        # Predict Q values for current and next states and restrict action space
        # NOTE: training=False for not using dropout layers.
        q_current = self._model(curr_state, training=False)
        q_next_online = self._model(next_state, training=False) + next_action_mask
        q_next_target = self._target_model(next_state, training=False) + next_action_mask
        
        # Double DQN next q computation
        # Note: use online table to get action with highest q value for next state
        # Then, use the target table q table to compute q value for next state
        a_star = tf.argmax(q_next_online, axis=1, output_type=tf.int32)
        batch_indices = tf.range(self.batch_size, dtype=tf.int32)
        max_q_next = tf.gather_nd(q_next_target, tf.stack([batch_indices, a_star], axis=1))
        
        # Compute observed returns using Bellman equation
        returns = rewards + (self._gamma * max_q_next)
        
        # Create targets by updating only the taken actions' q values
        target_actions = tf.stack([batch_indices, tf.convert_to_tensor(self.actions, dtype=tf.int32)], axis=1)
        targets = tf.tensor_scatter_nd_update(q_current, target_actions, returns)

        loss = self._model.train_on_batch(curr_state, targets)
        self._update_target_network(tau=0.001)

        return float(loss)
    
    def _compute_loss(self, curr_state, curr_ef, prev_pos, action, reward, next_state, next_ef, curr_pos) -> float:
        # Prepare states
        curr_input = self._get_states(curr_state, [prev_pos] + curr_ef)
        next_input = self._get_states(next_state, [curr_pos] + next_ef)
        
        # Get mask for next action
        next_action_mask = self._get_mask(curr_pos)
        
        # Forward Passes
        q_current = self._model(curr_input, training=False)
        q_next_online = self._model(next_input, training=False) + next_action_mask
        q_next_target = self._target_model(next_input, training=False) + next_action_mask
        
        # Double DQN Logic
        a_star = tf.argmax(q_next_online[0], output_type=tf.int32)
        max_q_next = q_next_target[0, a_star]
        
        # Compute Bellman Target (Scalar)
        target_value = reward + (self._gamma * max_q_next)
        
        # Construct Full Target Vector
        q_target = q_current.numpy() 
        q_target[0, action] = float(target_value)
        
        # Compute Loss based on the model's loss function
        loss_val = self._model.loss(q_target, q_current)
        
        return float(loss_val)
    
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
        tc = 0 if action == DRLAgent.A_HOLD else 2.5e-3
        
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
            
    
    def _DSR(self, Rt, A_tm1, B_tm1, tc, eps: float = 1e-6) -> float:
        # Discount the returns by the transaction cost
        Rtd = Rt - tc
        
        # Compute delta values
        dAt = Rtd - A_tm1
        dBt = Rtd ** 2 - B_tm1

        num = B_tm1 * dAt - 0.5 * A_tm1 * dBt
        denom = max(B_tm1 - A_tm1 ** 2, 0.0) ** 1.5 + eps

        return np.clip(num / denom, -3.0, 3.0).astype(float)
    
    def _PnL(self, Rt, tc: float) -> float:
        # Discount the returns by the transaction cost
        Rtd = Rt - tc
        
        # Simple profit and loss reward
        return np.log1p(Rtd)

    
    def _DDDR(self, Rt, A_tm1, DD_tm1, tc, eps: float=1e-6) -> float:
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
                
        return np.clip(num / denom, -3.0, 3.0).astype(float)
    
    def _init_reward_params(self, Rts: list[float]) -> float | None:
        # Rts: List of returns to have a hot start for moments.
        # This is necessary for value stabilization
        # Returns any values needed for computing cum. reward and more..
        eps = 1e-6
        R0 = None
        
        # Ensure there are enough data points for initialization
        if len(Rts) < 2:
            raise ValueError('Not enough return data points to initialize reward parameters')
        
        # Convert Rts to numpy array for easier processing
        _Rts = np.array(Rts, dtype=np.float32)
        
        match self.reward_type:
            case 'DSR':
                # Set the Eta using standard EMA formula 2 / (W + 1)
                self._r_eta = 2 / (self._window_size + 1)
                
                # Compute initial moments
                self._r_A_tm1 = np.mean(_Rts, dtype=float)
                self._r_B_tm1 = np.mean(_Rts ** 2, dtype=float)
                    
                # Compute the initial sharpe ratio.
                std = max(self._r_B_tm1 - self._r_A_tm1 ** 2, 0.0) ** 0.5
                R0 = self._r_A_tm1 / (std + eps)
            case 'DDDR':
                # Set the Eta using standard EMA formula 2 / (W + 1)
                self._r_eta = 2 / (self._window_size + 1)
                
                # Compute the first moment
                self._r_A_tm1 = np.mean(_Rts, dtype=float)
                
                # Compute the negative returns and its downside deviation
                n_Rts = _Rts[_Rts < 0]
                
                if len(n_Rts) > 0:
                    self._r_DD2_tm1 = np.mean(n_Rts ** 2, dtype=float)
                else:
                    # Fallback if no negative returns in the window
                    self._r_DD2_tm1 = eps
                
                # Compute the initial downside deviation ratio. Doesn't account transaction cost
                R0 = self._r_A_tm1 / (self._r_DD2_tm1 ** 0.5 + eps)
            
        return R0
        
    def _get_reward_computes(self) -> dict[str, float]:
        match self.reward_type:
            case 'DSR':
                return {'A_tm1': self._r_A_tm1, 'B_tm1': self._r_B_tm1}
            case 'DDDR':
                return {'A_tm1': self._r_A_tm1, 'DD_tm1': self._r_DD2_tm1 ** 0.5}
            case _:
                return {}
    
    def _set_reward_computes(self, params: dict[str, float]) -> None:
        # WARNING: This method is dangerous to use as it can desynchronize
        # the reward compute variables from the actual returns seen so far.
        # Use with caution.
        match self.reward_type:
            case 'DSR':
                self._r_A_tm1 = params['A_tm1']
                self._r_B_tm1 = params['B_tm1']
            case 'DDDR':
                self._r_A_tm1 = params['A_tm1']
                self._r_DD2_tm1 = params['DD_tm1'] ** 2
            case _:
                pass
    
    def _update_reward_computes(self, Rt) -> dict[str, float]:
        match self.reward_type:
            case 'DSR':
                self._r_A_tm1 += self._r_eta * (Rt - self._r_A_tm1)
                self._r_B_tm1 += self._r_eta * (Rt ** 2 - self._r_B_tm1)
            case 'DDDR':
                self._r_A_tm1 += self._r_eta * (Rt - self._r_A_tm1)
                self._r_DD2_tm1 += self._r_eta * (min(Rt, 0) ** 2 - self._r_DD2_tm1)
            case _:
                pass
        return self._get_reward_computes()

    def train(self, state_loader: EpisodeStateLoader, episode_ids: list[int], train_config: dict[str, Any]) -> None:
        # Make req. directories if not exist
        os.makedirs(train_config['model_dir'], exist_ok=True)
        os.makedirs(train_config['plots_dir'], exist_ok=True)
        os.makedirs(train_config['logs_dir'], exist_ok=True)
        
        # Init for logging and plotting purposes
        train_losses = []
        val_losses = []
        logs_by_episode = {}
        eps_start = []
        eps_curr = []
        eps_end = []
        
        # Get all tickers symbols
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
            # episode_reward_reqs are values that are used for visualization purposes
            # see _visualize_rewards()
            episode_reward_reqs = np.zeros(len(all_tickers))
            # Store the reward values per time step per ticker
            episode_rewards = np.zeros((len(all_tickers), L - t0 - 1))
            
            # Iterate tickers, training sequentially
            for ti, ticker in enumerate(tqdm(all_tickers, desc=f'Training episode {e}', ncols=100)):

                curr_state = state_loader.get_state_matrix('train', e, ticker, t0, self._window_size)
                prev_pos = DRLAgent.OUT_TRADE
                
                # Initalize the reward computes for the ticker
                # NOTE: Some reward functions (like DSR, DDDR require) need
                # a non-zero inital values of moments (i.e, a hot start)
                Rts = [state_loader.get_reward_computes('train', e, ticker, i)['1DFRet'] for i in range(t0)]
                R0 = self._init_reward_params(Rts)
                episode_reward_reqs[ti] = R0
                    
                for t in range(t0, L - 1):
                    # Get the reward computes that are included in the state representation
                    reward_computes = self._get_reward_computes()
                    # Store the current extra features (ef) used in state rep.
                    curr_ef = list(reward_computes.values())
                    
                    # Derive action based on eps-greedy policy
                    # NOTE: Here, extra features are some computes that are used
                    # to compute the reward. To make a valid MDP, there variables
                    # are included in the state representation
                    action, _ = self._act(curr_state, prev_pos, curr_ef)
                    
                    # Compute the current trade position based on new action
                    if prev_pos == DRLAgent.OUT_TRADE and action == DRLAgent.A_BUY:
                        curr_pos = DRLAgent.IN_TRADE
                    elif prev_pos == DRLAgent.IN_TRADE and action == DRLAgent.A_SELL:
                        curr_pos = DRLAgent.OUT_TRADE
                    else:
                        curr_pos = prev_pos
                    
                    # Append the return value for reward calculation
                    fRt = state_loader.get_reward_computes('train', e, ticker, t)['1DFRet']
                    Rt = fRt if curr_pos == DRLAgent.IN_TRADE else 0
                    reward_computes['Rt'] = Rt
                    
                    # Compute and store the reward value for the state-action pair
                    reward = self._compute_reward(reward_computes, action)
                    episode_rewards[ti, t - t0] = reward

                    # Get the next state
                    next_state = state_loader.get_state_matrix('train', e, ticker, t + 1, self._window_size)

                    # Update the reward compute variables and store the computes features
                    next_ef = list(self._update_reward_computes(Rt).values())
                    
                    # store transition
                    self._memory.append((curr_state, curr_ef, prev_pos, action, reward, next_state, next_ef, curr_pos))

                    # Update env steps taken
                    env_steps += 1
                    
                    # train from replay if enough samples; accumulate training loss for this group
                    if len(self._memory) >= self.replay_start_size:
                        # Train every train_interval steps
                        if env_steps % self.train_interval == 0:
                            loss = self._exp_replay()
                            train_loss += loss
                                
                            # Decay epsilon slowly until min
                            if self._epsilon > self._epsilon_min:
                                self._epsilon *= self._epsilon_decay
                    
                    # Advance to the next state
                    curr_state = next_state
                    prev_pos = curr_pos

            # Plot reward diagostics
            self._plot_rewards(episode_reward_reqs, episode_rewards, f'Epsiode {e}: Reward Visualization')
            
            # Run validation on this episode's validation set
            val_result = self._run_validation(state_loader, e, all_tickers)
            
            # Print the validation summary
            print(f'Episode {e} validation summary:')
            print(f"Train loss: {train_loss:.4f}, Val loss: {val_result['total_loss']:.4f}, Total val trades: {val_result['total_trades']}, Hit rate: {val_result['hit_rate']:.2f}")
            print(f"Trade Duration: {val_result['trade_duration']:.2f}, Total PnL: {val_result['total_pnl']:.2f}, Profit Factor: {val_result['profit_factor']:.3f}")
            print(f"Force End Trade Count: {val_result['force_end_trades']}, Force End PnL: {val_result['force_end_pnl']:.2f}")
            
            # Append the train_losses and val_losses
            train_losses.append(train_loss)
            val_losses.append(val_result['total_loss'])
            
            # Boost the epsilon a bit for next episode (as every episode has diff regimes)
            epsilon_end = self._epsilon
            self._epsilon = epsilon_end + self._epsilon_boost_factor * (epsilon_start - epsilon_end)
            
            # Append epsilon values for plotting
            eps_start.append(epsilon_start)
            eps_curr.append(self._epsilon)
            eps_end.append(epsilon_end)
            
            # Store logs for this episode
            logs_by_episode[e] = {
                'train_loss': train_loss,
                'val_results': val_result,
                'epsilon_start': epsilon_start,
                'epsilon_current': self._epsilon,
                'epsilon_end': epsilon_end
            }
            
        # Plot all the training and validation losses
        self._plot_losses(train_losses, val_losses, state_loader, fname=os.path.join(train_config['plots_dir'], 'episode_losses.png'))
        
        # Plot the epsilon decay
        eps_fname = os.path.join(train_config['plots_dir'], 'epsilon_decay.png')
        self._plot_epsilon_decay(eps_start, eps_curr, eps_end, fname=eps_fname)
        
        # Save model checkpoint with the current date and time
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._model.save(os.path.join(train_config['model_dir'], f'model_{date_str}.keras'))
        
        # Save logs to a json file
        with open(os.path.join(train_config['logs_dir'], f'train_logs_{date_str}.json'), 'w') as f:
            json.dump(logs_by_episode, f, indent=2)

    def _plot_rewards(self, reward_reqs, reward_data, plot_name: str, remove_outliers: bool=True) -> None:
        # Reward reqs are used by some reward function to compute cumulative reward
        # Create plot with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1]})
        fig.suptitle(plot_name, fontsize=12)
        
        # Left plot: Cumulative sum of reward (i.e, approximations of e.g., Sharpe ratio)
        # First compute the cumulative rewards based on the reward type
        if self.reward_type in {'DSR', 'DDDR'}:
            # Formula: S_t = S_0 + eta * Sum(D_t)
            # We reshape reward_reqs to (N, 1) to broadcast across time axis
            cum_sums = np.cumsum(reward_data, axis=1)
            all_cum_rewards = reward_reqs[:, np.newaxis] + (self._r_eta * cum_sums)
        elif self.reward_type in {'PNL'}:
            # Formula: exp(Sum(log_returns))
            all_cum_rewards = np.exp(np.cumsum(reward_data, axis=1))
        else:
            # Fallback for raw summation if other types added later
            all_cum_rewards = np.cumsum(reward_data, axis=1)
        y_label = f'Cumulative Reward {self.reward_type}'
        
        # Compute Statistics across tickers (mean, std)
        mean_r_traj = np.mean(all_cum_rewards, axis=0)
        std_r_traj = np.std(all_cum_rewards, axis=0)
        
        # Plot the Mean Line and shaded standard deviation
        time_steps = np.arange(reward_data.shape[1])
        ax1.plot(time_steps, mean_r_traj, color='#1f77b4', linewidth=2, label='Mean Performance')
        ax1.fill_between(time_steps, mean_r_traj - std_r_traj, mean_r_traj + std_r_traj, color='#1f77b4', alpha=0.2, label='±1 Std. Dev.')

        ax1.set_title('Overall Cumulative Reward Growth')
        ax1.set_xlabel('Time steps')
        ax1.set_ylabel(y_label)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Right Plot: Distribution of the receieved rewards
        rewards_flat = reward_data.ravel()
        
        # IQR-based clipping to avoid outliers in histogram, mean, median
        Q1 = np.percentile(rewards_flat, 25)
        Q3 = np.percentile(rewards_flat, 75)
        # Calculate upper and lower bounds
        IQR = Q3 - Q1
        low = Q1 - 1.5 * IQR
        high = Q3 + 1.5 * IQR
        if remove_outliers and IQR > 0:
            rewards_hist = rewards_flat[(rewards_flat >= low) & (rewards_flat <= high)]
            print(f'Removed {len(rewards_flat) - len(rewards_hist)} outliers from histogram.')
        else:
            rewards_hist = rewards_flat
        
        mean_reward = np.mean(rewards_hist)
        median_reward = np.median(rewards_hist)
        # Always include the outlier when computing skewness and kurtosis
        skew_val = stats.skew(rewards_flat)
        kurt_val = stats.kurtosis(rewards_flat)
        
        # Histogram without outliers
        ax2.hist(rewards_hist, bins=30, alpha=0.6, edgecolor='black', density=True)
        # Include a KDE line with outliers
        sns.kdeplot(rewards_flat, ax=ax2, label='KDE', color='orange')
        # However, limit x-axis if outliers are removed
        if remove_outliers and IQR > 0:
            ax2.set_xlim(left=low, right=high)
        # Lines to show mean and the median
        ax2.axvline(x=mean_reward, linestyle='--', label=f'μ ({mean_reward:.2f})')
        ax2.axvline(x=median_reward, linestyle=':', label=f'Med. ({median_reward:.2f})')
        
        stats_text = f'Skew: {skew_val:.2f}\nKurt: {kurt_val:.2f}'
        ax2.text(0.95, 0.95, stats_text,
                 transform=ax2.transAxes,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax2.legend(loc='upper left')
        ax2.set_title('Reward Distribution' + (' (No Outliers)' if remove_outliers else ''))
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Density')
        
        plt.tight_layout()
        plt.show()

    
    def _run_validation(self, state_loader: EpisodeStateLoader, episode_id: int, tickers: list[str]) -> dict[str, int | float]:
        # The agent does no exploration, only exploitation
        # Define metrics to track for validation cycle
        total_loss = 0.0
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
            curr_state = state_loader.get_state_matrix('validate', episode_id, ticker, t0, self._window_size)
            prev_pos = DRLAgent.OUT_TRADE
            entry_price = None

            # Initalize the reward computes for the ticker
            # NOTE: Some reward functions (like DSR, DDDR require) need
            # a non-zero inital values of moments (i.e, a hot start)
            Rts = [state_loader.get_reward_computes('train', episode_id, ticker, i)['1DFRet'] for i in range(t0)]
            self._init_reward_params(Rts)
            
            for t in range(t0, L - 1):
                # Get the reward computes that are included in the state representation
                reward_computes = self._get_reward_computes()
                # Store the current extra features (ef) for deciding action
                curr_ef = list(reward_computes.values())
                
                # training=False turns off exploration
                action, _ = self._act(curr_state, prev_pos, curr_ef, training=False)
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
                next_ef = list(self._update_reward_computes(Rt).values())
                
                # Compute the validation error/loss
                total_loss += self._compute_loss(curr_state, curr_ef, prev_pos, action, reward, next_state, next_ef, curr_pos)
                
                # Advance to the next state
                curr_state =  next_state
                prev_pos = curr_pos

        # WARNING: Metrics consider forcefully ended trades which could skew performance
        total_pnl = gross_profit + gross_loss
        hit_rate = winning_trade_count / max(total_trades, 1)
        trade_duration = in_trade_days / max(total_trades, 1)
        profit_factor = gross_profit / max(abs(gross_loss), 1e-12)
        
        return {
            'total_loss': total_loss,
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
    
    def _plot_losses(self, train_losses, val_losses: list[float], state_loader: EpisodeStateLoader, fname: str | None = None, show: bool = True):
        if len(train_losses) != len(val_losses):
            raise ValueError('Train and validation losses must have the same length')
        
        # We first need to compute the scaled training and validation losses.
        # Why? Training windows are expanding, naturally having higher losses.
        # So, we scale them based on window size to have a fair comparison.
        # However, this is a naive approach and may not be perfect.
        train_losses_scaled = []
        val_losses_scaled = []
        for i in range(len(train_losses)):
            train_w_length = state_loader.get_episode_len('train', i)
            val_w_length = state_loader.get_episode_len('validate', i)
            
            # If first episode, reduce the replay start size from training window
            # as these timesteps are not used for training immediately
            if i == 0:
                train_w_length -= self.replay_start_size
                
            # Compute the scaled losses
            train_losses_scaled.append(train_losses[i] / train_w_length)
            val_losses_scaled.append(val_losses[i] / val_w_length)

        # Create the plot and axes
        fig, ax1 = plt.subplots(figsize=(10, 4))
        x = np.arange(1, len(train_losses) + 1)
        
        # Left axis: training loss
        ax1.plot(x, train_losses_scaled, marker='o', linewidth=2, color='tab:blue',
                label='Train loss')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Train loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, linestyle='--', alpha=0.3)

        # Right axis: validation loss
        ax2 = ax1.twinx()
        ax2.plot(x, val_losses_scaled, marker='s', linewidth=2, color='tab:orange',
                label='Validation loss')
        ax2.set_ylabel('Validation loss', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Combine legends
        lines, labels = [], []
        for ax in [ax1, ax2]:
            l, lab = ax.get_legend_handles_labels()
            lines.extend(l)
            labels.extend(lab)
        ax1.legend(lines, labels, loc='upper right')

        plt.title('Training vs. Validation Loss [Scaled by WFV Window Size]')
        fig.tight_layout()

        if fname:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            plt.savefig(fname, dpi=150)

        if show:
            plt.show()
        else:
            plt.close()

    def _get_predict_dict(self, a: int, q: float) -> dict:
        # A = {0: buy, 1: hold, 2: sell}
        action_map = {
            DRLAgent.A_BUY: 'buy',
            DRLAgent.A_HOLD: 'hold',
            DRLAgent.A_SELL: 'sell'
        }
        
        return {'action': action_map[a], 'q_value': q}
    
    def test(self, state_loader: EpisodeStateLoader, episode_id: int, test_config: dict[str, Any]):
        # NOTE: This function is depreciated. Please use Event-Driven backtesting module.
        warnings.warn(
            'test() is deprecated as it produces static signals. Use Event-Driven backtesting module instead.',
                      DeprecationWarning,
                      stacklevel=2)
        
        all_tickers = state_loader.get_all_tickers()

        # Keep a single ordered list of tickers and parallel lists of series for safe alignment
        col_keys = [] # ['AAPL', 'MSFT', ...] in deterministic order
        sig_series_list = [] # [Series-of-dicts, ...]
        px_series_list  = [] # [Series-of-floats, ...]
        
        L = state_loader.get_episode_len('test', episode_id)
        
        # Get the index for the signals and prices dataframes
        df_idx = state_loader.get_test_dates(episode_id)
        
        for ticker in tqdm(all_tickers, desc=f'Testing episode {episode_id}', ncols=100):
            # Init the start index
            t0 = self._window_size - 1
            
            # Initalize the reward computes for the ticker
            # NOTE: Some reward functions (like DSR, DDDR require) need
            # a non-zero inital values of moments (i.e, a hot start)
            Rts = [state_loader.get_reward_computes('test', episode_id, ticker, i)['1DFRet'] for i in range(t0)]
            self._init_reward_params(Rts)
            
            curr_state = state_loader.get_state_matrix('test', episode_id, ticker, t0, self._window_size)
            prev_pos = DRLAgent.OUT_TRADE

            # Allocate containers (length L to match idx)
            sig_cells = [None] * L
            close_px  = np.empty(L, dtype=np.float32)

            # main test loop
            for t in range(t0, L - 1):
                # Get the reward computes that are included in the state representation
                reward_computes = self._get_reward_computes()
                # Store the current extra features (ef) for deciding action
                curr_ef = list(reward_computes.values())
                
                # Get the current close price
                curr_close = state_loader.get_state_OHLCV('test', episode_id, ticker, t)['Close']
                close_px[t] = curr_close

                # Derive action based on greedy policy
                action, soft_q = self._act(curr_state, prev_pos, curr_ef, training=False)
                sig_cells[t] = self._get_predict_dict(action, soft_q) # type: ignore

                # Compute the current trade position based on new action
                if prev_pos == DRLAgent.OUT_TRADE and action == DRLAgent.A_BUY:
                    curr_pos = DRLAgent.IN_TRADE
                elif prev_pos == DRLAgent.IN_TRADE and action == DRLAgent.A_SELL:
                    curr_pos = DRLAgent.OUT_TRADE
                else:
                    curr_pos = prev_pos
                
                # Update the reward computes for the next iteration
                fRt = state_loader.get_reward_computes('test', episode_id, ticker, t)['1DFRet']
                Rt = fRt if curr_pos == DRLAgent.IN_TRADE else 0
                self._update_reward_computes(Rt)
                
                # Advance to the next state and update prev_pos
                next_state = state_loader.get_state_matrix('test', episode_id, ticker, t + 1, self._window_size)
                curr_state = next_state
                prev_pos = curr_pos

            # Final row (t = L-1): record price; no new decision possible so force hold-out or sell
            close_px[L - 1] = state_loader.get_state_OHLCV('test', episode_id, ticker, L - 1)['Close']
            if sig_cells[L - 1] is None:
                if prev_pos == DRLAgent.OUT_TRADE:
                    # Hold from taking a trade
                    sig_cells[L - 1] = self._get_predict_dict(DRLAgent.A_HOLD, 1.0) # type: ignore
                else:
                    # Close the trade
                    sig_cells[L - 1] = self._get_predict_dict(DRLAgent.A_SELL, 1.0) # type: ignore

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