import os
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from typing import Any
from tensorflow import keras
from tensorflow.keras import layers, Input, optimizers
from collections import deque
from tqdm import tqdm

from .state import EpisodeStateLoader

@keras.utils.register_keras_serializable()
# Define DQN Model Architecture
class DualBranchDQN(keras.Model):
    def __init__(self, motif_state_shape: tuple[int, int], context_state_size: int, action_size: int, learning_rate: float) -> None:
        super().__init__()

        # Motif Branch for finding candle patterns using Conv1D
        motif_input = Input(shape=motif_state_shape, name="motif_input")
        motif_conv_layer = layers.Conv1D(32, 3, padding="same", activation="relu")(motif_input)
        motif_conv_layer = layers.Conv1D(32, 3, padding="same", activation="relu")(motif_conv_layer)
        motif_out = layers.GlobalAveragePooling1D(name="motif_output")(motif_conv_layer)

        # Context Branch for finding regimes
        context_input = Input(shape=(context_state_size,), name="context_input")
        context_hid_layer = layers.Dense(64, activation="relu")(context_input)
        context_out = layers.Dense(32, activation="relu", name="context_output")(context_hid_layer)
        
        # Late fusion of both the motif and context branches
        fused_branch = layers.Concatenate(name="late_fusion")([motif_out, context_out])
        fused_branch = layers.Dense(128, activation="relu")(fused_branch)
        fused_branch = layers.Dense(64, activation="relu")(fused_branch)
        fused_branch = layers.LayerNormalization()(fused_branch)
        Q = layers.Dense(action_size, name="q_values")(fused_branch)

        model_input = {"motif_input": motif_input, "context_input": context_input}
        self._model = keras.Model(inputs=model_input, outputs=Q, name="DualBranchDQN")
        self._model.compile(loss="mse", optimizer=optimizers.Adam(learning_rate, clipnorm=1.0))

    def get_model(self):
        return self._model

class RLAgent:
    def __init__(self, agent_config, test_mode: int=False, model_name: str='') -> None:
        # Store state and action representation configurations
        self._window_size = agent_config['state_matrix_window']
        self._num_motif_feat = agent_config['num_motif_features']
        self._num_context_feat = agent_config['num_context_features']
        
        # Hold: 0, Buy: 1, Sell: 2
        self._action_size = 3

        # Define memory to store (state, action, reward, new_state) pairs for exp. replay
        buffer_length = agent_config.get('memory_buffer_len', 10000)
        self._memory = deque(maxlen=buffer_length)

        # Define inventory to hold trades
        self._inventory = []

        self._test_mode = test_mode
        self._model_name = model_name

        # Define parameters for behaviour policy and temporal learning
        self._gamma = agent_config.get('gamma', 0.95)  # discount factor
        self._epsilon = agent_config.get('epsilon_start', 1.0)
        self._epsilon_min = agent_config.get('epsilon_min', 0.01)
        n_updates = agent_config.get('decay_updates', 25000) # No. of updates to minimum epsilon
        self._epsilon_decay = (self._epsilon_min / self._epsilon) ** (1.0 / n_updates)

        # Load model or define new
        learning_rate = agent_config.get('learning_rate', 1e-3)
        self._model = keras.models.load_model(model_name) if self._test_mode else self._init_model(learning_rate)
        
        # Init target network update and frequency
        self._init_target_network()
    
    def get_model(self) -> keras.Model:
        return self._model

    def _init_model(self, learning_rate) -> keras.Model:
        # Compute state shapes for the model
        motif_shape = (self._window_size, self._num_motif_feat)
        # Note: Add 1 to include the pos_t flag
        context_size = self._window_size * self._num_context_feat + 1

        # Init model
        dual_dqn = DualBranchDQN(motif_shape, context_size, self._action_size, learning_rate)

        return dual_dqn.get_model()

    def _init_target_network(self) -> None:
        # Make a structural clone and copy weights
        self._target_model = keras.models.clone_model(self._model)
        self._target_model.set_weights(self._model.get_weights())


    def update_target_network(self, tau: float = 1.0) -> None:
        # tau = 1.0 : hard update (copy weights exactly)
        # tau < 1.0 : soft/Polyak update (exponential moving average)
        
        online_weights = self._model.get_weights()
        target_weights = self._target_model.get_weights()

        new_weights = [
            tau * w_online + (1.0 - tau) * w_target
            for w_online, w_target in zip(online_weights, target_weights)]

        self._target_model.set_weights(new_weights)

    
    def _get_states(self, state_matrix: np.ndarray, prev_pos: int) -> dict[str, np.ndarray]:
        # Check the state matrix shape is correct
        total_features = self._num_motif_feat + self._num_context_feat
        if state_matrix.shape != (self._window_size, total_features):
            raise KeyError(f'Invalid state matrix shape: {state_matrix.shape}')
        
        # Define the motif input
        # Expand dims from [window, num_features] to [batch_size=1, window, num_features]
        motif_input = np.expand_dims(state_matrix[:, :self._num_motif_feat], axis=0).astype(np.float32)

        # Flatted the context input and combine with post_t
        context_flat = state_matrix[:, self._num_motif_feat:].astype(np.float32).reshape(1, -1)
        prev_pos_arr = np.array([[float(prev_pos)]], dtype=np.float32)
        context_input = np.concatenate([context_flat, prev_pos_arr], axis=1)

        return {'motif_input': motif_input, 'context_input': context_input}
    
    def get_q_values(self, state_matrix: np.ndarray, prev_pos: int) -> np.ndarray:
        # Predict q values through DQN
        model_input = self._get_states(state_matrix, prev_pos)
        q_values = self._model.predict(x=model_input, verbose=0)

        return q_values
    
    def fit_model(self, state_matrix: np.ndarray, prev_pos: int, target_q: np.ndarray) -> float:
        # Compute MSE loss between actual and target q values
        model_input = self._get_states(state_matrix, prev_pos)
        loss = self._model.fit(model_input, target_q, epochs=1, verbose=0)    

        return loss

    def act(self, state_matrix: np.ndarray, prev_pos: int) -> int:
        if prev_pos not in {0, 1}:
            raise ValueError(f'Invalid previous position: {prev_pos}')
        
        # Defines an epsilon-greedy behaviour policy
        # Pick random action epsilon times
        if not self._test_mode and np.random.random() < self._epsilon:
            return np.random.randint(self._action_size)

        # Pick action from DQN 1-epsilon times
        q_values = self.get_q_values(state_matrix, prev_pos)
        return int(np.argmax(q_values))

    def exp_replay(self, batch_size: int) -> float:
        if len(self._memory) < batch_size:
            return 0.0

        # Prefer uniform sampling over entire buffer (stability)
        idx = np.random.choice(len(self._memory), size=batch_size, replace=False)
        batch = [self._memory[i] for i in idx]

        B, W, M, C = batch_size, self._window_size, self._num_motif_feat, self._num_context_feat
        motif_batch = np.empty((B, W, M), dtype=np.float32)
        context_batch = np.empty((B, W*C + 1), dtype=np.float32)
        next_motif_batch = np.empty((B, W, M), dtype=np.float32)
        next_context_batch = np.empty((B, W*C + 1), dtype=np.float32)
        actions = np.empty((B,), dtype=np.int32)
        rewards = np.empty((B,), dtype=np.float32)
        dones = np.empty((B,), dtype=np.float32)

        for i, (state, prev_pos, action, reward, next_state, next_prev_pos, done) in enumerate(batch):
            # Extract state components for feeding to DQN braches
            motif = state[:, :M].astype(np.float32)
            ctx = state[:, M:].astype(np.float32).reshape(-1)
            motif_batch[i] = motif
            context_batch[i] = np.concatenate([ctx, np.array([prev_pos], dtype=np.float32)], axis=0)

            # If no next_state, set done flag to True
            if next_state is None:
                next_motif_batch[i] = motif
                next_context_batch[i] = context_batch[i]
                dones[i] = 1.0
            else:
                next_motif = next_state[:, :M].astype(np.float32)
                next_ctx   = next_state[:, M:].astype(np.float32).reshape(-1)  # [W*C]
                next_motif_batch[i]   = next_motif
                next_context_batch[i] = np.concatenate([next_ctx, np.array([next_prev_pos], dtype=np.float32)], axis=0)
                dones[i] = 1.0 if done else 0.0

            actions[i] = action
            rewards[i] = reward

        q_current = self._model.predict({"motif_input": motif_batch, "context_input": context_batch}, verbose=0)
        q_next_online = self._model.predict({"motif_input": next_motif_batch, "context_input": next_context_batch}, verbose=0)
        q_next_target = self._target_model.predict({"motif_input": next_motif_batch, "context_input": next_context_batch}, verbose=0)

        # Compute and update to the ideal or target q table
        # Note: use online table to get max action for next state
        # Then, use the target table q value for that next state and max action
        a_star = np.argmax(q_next_online, axis=1)
        max_q_next = q_next_target[np.arange(B), a_star].astype(np.float32)
        
        returns = rewards + (1.0 - dones) * (self._gamma * max_q_next)
        targets = q_current.copy()
        targets[np.arange(B), actions] = returns

        loss = self._model.train_on_batch({"motif_input": motif_batch, "context_input": context_batch}, targets)
        self.update_target_network(tau=0.005)

        return float(loss)


    @staticmethod
    def calculate_reward(prev_pos: int, action: int, curr_price: float, next_price: float) -> tuple[float, int]:
        # Note: A buy signal when prev_pos is 1 is equivlent to hold
        # A sell signal when prev_pos is 0 is equivlent to hold
        # Update position from action
        if action == 0:
            pos_t = prev_pos
        elif action == 1:
            pos_t = 1
        elif action == 2:
            pos_t = 0

        log_ret = np.log(next_price / max(curr_price, 1e-12))
        trade_cost = 0.0005 if pos_t != prev_pos else 0.0
        reward = pos_t * log_ret - trade_cost
        return reward, pos_t
    
    def train(self, state_loader: EpisodeStateLoader, episode_ids: list[int], train_config: dict[str, Any]) -> dict[str, Any]:
        # Make req. directories if not exist
        os.makedirs(train_config['model_dir'], exist_ok=True)
        os.makedirs(train_config['plots_dir'], exist_ok=True)
        os.makedirs(train_config['logs_dir'], exist_ok=True)

        logs_by_episode = {}

        tickers_all = state_loader.get_all_tickers()

        for e in episode_ids:

            # Compute the total loss across the whole episode
            train_loss = 0.0

            # Track env steps
            env_steps = 0
            
            # Iterate tickers, training sequentially
            for ticker in tqdm(tickers_all, desc=f'Training episode {e}', ncols=100):
                L = state_loader.get_episode_len('train', e, ticker)
                if L < self._window_size + 1:
                    # skip tickers that don't have enough data for a window and next step
                    # Not the case for the current episode configuration file
                    continue

                # Train on this ticker for the episode (standard rollout with replay)
                t0 = self._window_size - 1
                state = state_loader.get_state_matrix('train', e, ticker, t0, self._window_size)
                prev_pos = 0

                for t in range(t0, L - 1):
                    # Find action based on behaviour policy
                    action = self.act(state, prev_pos)

                    curr_price = float(state_loader.get_state_OHLCV('train', e, ticker, t)[3])
                    next_price = float(state_loader.get_state_OHLCV('train', e, ticker, t + 1)[3])

                    reward, pos_t = RLAgent.calculate_reward(prev_pos, action, curr_price, next_price)

                    next_state = state_loader.get_state_matrix('train', e, ticker, t + 1, self._window_size)
                    done = (t == L - 2)

                    # store transition
                    self._memory.append((state, prev_pos, action, reward, next_state, pos_t, done))

                    # Update env steps taken
                    env_steps += 1
                    
                    # train from replay if enough samples; accumulate training loss for this group
                    if len(self._memory) >= train_config.get('replay_start_size', 5000):
                        # Train every train_interval steps
                        if env_steps % train_config.get('train_interval', 1) == 0:
                            for _ in range(train_config.get('gradient_step_repeat', 1)):
                                loss = self.exp_replay(train_config.get('batch_size', 256))
                                train_loss += loss
                                
                                # Decay epsilon slowly until min
                                if self._epsilon > self._epsilon_min and not self._test_mode:
                                    self._epsilon *= self._epsilon_decay 
                    
                    # Advance to the next state
                    state = next_state
                    prev_pos = pos_t

            # Run validation on this episode's validation set
            val_results = self._run_validation(state_loader, e, tickers_all)
            val_loss = -val_results['sum_reward']
            print(f"Episode {e} summary: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Total val trades: {val_results['total_trades']}, Hit rate: {val_results['hit_rate']:.3f}")
            
            # Store logs for this episode
            logs_by_episode[e] = {
                "train_loss": train_loss,
                "val_results": val_results,
                "epsilon": self._epsilon
            }
            
        # Plot all the training and validation losses
        train_losses = [logs_by_episode[ep]["train_loss"] for ep in episode_ids]
        val_losses = [-logs_by_episode[ep]["val_results"]["sum_reward"] for ep in episode_ids]
        self._plot_losses(train_losses, val_losses, fname=os.path.join(train_config['plots_dir'], 'train_losses.png'))
        
        # Also plot the epsilon decay
        epsilons = [logs_by_episode[ep]["epsilon"] for ep in episode_ids] 
        self._plot_epsilon_decay(epsilons, fname=os.path.join(train_config['plots_dir'], 'epsilon_decay.png'))
        
        # Save model checkpoint with the current date and time
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._model.save(os.path.join(train_config['model_dir'], f"model_{date_str}.keras"))
        
        # Save logs to a json file
        with open(os.path.join(train_config['logs_dir'], f"train_logs_{date_str}.json"), 'w') as f:
            json.dump(logs_by_episode, f, indent=2)

        return logs_by_episode


    def _run_validation(self, state_loader: EpisodeStateLoader, episode_id: int, tickers: list[str]):
        # switch to test mode for validation
        prev_test_mode = self._test_mode
        self._test_mode = True

        total_reward = 0.0
        total_trades = 0
        wins = 0
        losses = 0

        for ticker in tickers:
            L = state_loader.get_episode_len('validate', episode_id, ticker)
            if L < self._window_size + 1:
                continue

            t0 = self._window_size - 1
            state = state_loader.get_state_matrix('validate', episode_id, ticker, t0, self._window_size)
            prev_pos = 0
            entry_price = None

            for t in range(t0, L - 1):
                action = self.act(state, prev_pos)
                curr_price = float(state_loader.get_state_OHLCV('validate', episode_id, ticker, t)[3])
                next_price = float(state_loader.get_state_OHLCV('validate', episode_id, ticker, t + 1)[3])

                r_t, pos_t = RLAgent.calculate_reward(prev_pos, action, curr_price, next_price)
                total_reward += r_t

                # simple trade bookkeeping
                if prev_pos == 0 and pos_t == 1:
                    entry_price = curr_price
                    total_trades += 1
                if prev_pos == 1 and pos_t == 0 and entry_price is not None:
                    trade_pnl = curr_price - entry_price
                    if trade_pnl >= 0:
                        wins += 1
                    else:
                        losses += 1
                    entry_price = None

                next_state = state_loader.get_state_matrix('validate', episode_id, ticker, t + 1, self._window_size)
                
                # Advance to the next state
                state =  next_state
                prev_pos = pos_t

        self._test_mode = prev_test_mode

        hit_rate = wins / max(wins + losses, 1)
        return {
            "episode": episode_id,
            "sum_reward": float(total_reward),
            "total_trades": int(total_trades),
            "hit_rate": float(hit_rate),
        }

    def _plot_epsilon_decay(self, epsilons: list[float], fname: str | None = None, show: bool = True):
        x = np.arange(1, len(epsilons) + 1)

        plt.figure(figsize=(8, 4))
        plt.plot(x, epsilons, marker='o', linewidth=2, color='tab:blue')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay over Episodes')
        plt.grid(True, linestyle='--', alpha=0.3)

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