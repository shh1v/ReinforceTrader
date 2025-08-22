import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Any
from tensorflow import keras
from tensorflow.keras import Input, layers, optimizers
from collections import deque
from tqdm import tqdm

from .state import EpisodeStateLoader

@keras.utils.register_keras_serializable()
# Define DQN Model Architecture
class DualBranchDQN(keras.Model):
    def __init__(self, motif_state_shape: tuple[int, int], context_state_size: int, action_size: int):
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
        self._model.compile(loss="mse", optimizer=optimizers.Adam(1e-3))

    def get_model(self):
        return self._model

class RLAgent:
    def __init__(self, window_size: int, num_features: tuple[int, int], test_mode: int=False, model_name: str=''):
        # Store state and action representation configurations
        self._window_size = window_size
        self._num_motif_feat, self._num_context_feat = num_features
        self._action_size = 3 # Hold: 0, Buy: 1, Sell: 2

        # Define memory to store (state, action, reward, new_state) pairs for exp. replay
        self._memory = deque(maxlen=1000)

        # Define inventory to hold trades
        self._inventory = []

        self._test_mode = test_mode
        self._model_name = model_name

        # Define parameters for behaviour policy and temporal learning
        self._gamma = 0.95
        self._epsilon = 1.0
        self._epsilon_min = 0.01
        self._epsilon_decay = 0.995

        # Load model or define new
        self._model = keras.models.load_model(model_name) if self._test_mode else self._init_model()
        
        # Init target network update and frequency
        self._init_target_network()
    
    def get_model(self):
        return self._model

    def _init_model(self) -> keras.Model:
        # Compute state shapes for the model
        motif_shape = (self._window_size, self._num_motif_feat)
        context_size = self._window_size * self._num_context_feat

        # Init model
        dual_dqn = DualBranchDQN(motif_shape, context_size, self._action_size)

        return dual_dqn.get_model()

    def _init_target_network(self):
        # Make a structural clone and copy weights
        self._target_model = keras.models.clone_model(self._model)
        self._target_model.set_weights(self._model.get_weights())


    def update_target_network(self, tau: float = 1.0):
        # tau = 1.0 : hard update (copy weights exactly)
        # tau < 1.0 : soft/Polyak update (exponential moving average)
        
        online_weights = self._model.get_weights()
        target_weights = self._target_model.get_weights()

        new_weights = [
            tau * w_online + (1.0 - tau) * w_target
            for w_online, w_target in zip(online_weights, target_weights)]

        self._target_model.set_weights(new_weights)

    
    def _get_states(self, state_matrix: np.ndarray) -> dict[str, np.ndarray]:
        # Check the state matrix shape is correct
        total_features = self._num_motif_feat + self._num_context_feat
        if state_matrix.shape != (self._window_size, total_features):
            raise KeyError(f'Invalid state matrix shape: {state_matrix.shape}')
        
        # Define the motif input
        # Expand dims from [window, num_features] to [batch_size=1, window, num_features]
        motif_input = np.expand_dims(state_matrix[:, :self._num_motif_feat], axis=0).astype(np.float32)

        # Flatted the context input to [1, window * num_context_features]
        context_input = state_matrix[:, self._num_motif_feat:].reshape(1, -1).astype(np.float32)

        return {'motif_input': motif_input, 'context_input': context_input}
    
    def get_q_values(self, state_matrix: np.ndarray) -> np.ndarray:
        # Predict q values through DQN
        model_input = self._get_states(state_matrix)
        q_values = self._model.predict(x=model_input, verbose=0)

        return q_values
    
    def fit_model(self, state_matrix: np.ndarray, target_q: np.ndarray):
        # Compute MSE loss between actual and target q values
        model_input = self._get_states(state_matrix)
        loss = self._model.fit(model_input, target_q, epochs=1, verbose=0)    

        return loss

    def act(self, state_matrix: np.ndarray) -> int:
        # Defines an epsilon-greedy behaviour policy
        # Pick random action epsilon times
        if not self._test_mode and np.random.random() < self._epsilon:
            return np.random.randint(self._action_size)

        # Pick action from DQN 1-epsilon times
        q_values = self.get_q_values(state_matrix)
        return int(np.argmax(q_values))

    def exp_replay(self, batch_size: int):
        if len(self._memory) < batch_size:
            return []

        # Derive batch from memory and shuffle
        memory_size = len(self._memory)
        idx = np.random.permutation(np.arange(memory_size - batch_size, memory_size))
        batch = [self._memory[i] for i in idx]

        # Pre-allocate arrays for simplicity
        B = batch_size
        W = self._window_size
        M = self._num_motif_feat
        C = self._num_context_feat

        motif_batch = np.empty((B, W, M), dtype=np.float32)
        context_batch = np.empty((B, W * C), dtype=np.float32)
        next_motif_batch = np.empty((B, W, M), dtype=np.float32)
        next_context_batch = np.empty((B, W * C), dtype=np.float32)

        actions = np.empty((B,), dtype=np.int32)
        rewards = np.empty((B,), dtype=np.float32)
        dones = np.empty((B,), dtype=np.float32)

        # Build batched tensors from transitions
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            # split motif / context for current state
            motif_state = state[:, :M]
            context_state = state[:, M:]
            motif_batch[i] = motif_state
            context_batch[i] = context_state.reshape(-1)

            # split motif / context for next state
            # Note: if None, reuse current & mark done. Although not possible
            if next_state is None:
                next_motif_batch[i] = motif_state
                next_context_batch[i] = context_state.reshape(-1)
                dones[i] = 1.0
            else:
                next_motif_state = next_state[:, :M]
                next_context_state = next_state[:, M:]
                next_motif_batch[i] = next_motif_state
                next_context_batch[i] = next_context_state.reshape(-1)
                dones[i] = 1.0 if done else 0.0

            actions[i] = action
            rewards[i] = reward

        # Forward passes on the whole batch. q_current: [B, A]
        q_current = self._model.predict(
            {"motif_input": motif_batch, "context_input": context_batch},
            verbose=0)

        # Use target model to stabilize learning. q_next: [B, A]
        q_next = self._target_model.predict(
            {"motif_input": next_motif_batch, "context_input": next_context_batch},
            verbose=0)

        # Use Bellman equation to compute targets for our q_current
        max_q_next = np.max(q_next, axis=1).astype(np.float32)  # [B]
        targets = q_current.copy()
        targets[:, actions] = rewards + (1.0 - dones) * (self._gamma * max_q_next)

        # Train the model on the whole batch
        loss = self._model.train_on_batch(
            {"motif_input": motif_batch, "context_input": context_batch},
            targets)

        # Update target network using polyak update
        self.update_target_network(tau=0.005)

        # epsilon decay (outside GPU path)
        if self._epsilon > self._epsilon_min and not self._test_mode:
            self._epsilon *= self._epsilon_decay
        
        # Keras returns the scaler loss
        return float(loss)

    
    def _compute_reward(self, prev_pos: int, action: int, curr_price: float, next_price: float) -> tuple[float, int]:
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
    
    def train(self, esl: EpisodeStateLoader, episode_ids: list[int], config: dict[str, Any]) -> dict[str, Any]:
        # Make req. directories if not exist
        os.makedirs(config['model_dir'], exist_ok=True)
        os.makedirs(config['plots_dir'], exist_ok=True)

        logs_by_episode = {}

        tickers_all = esl.get_all_tickers()

        for e in episode_ids:
            # Per-episode per-group stores of training performance
            # list of train loss (floats), one per group
            group_train_losses = []
            # list of validation results (dict), one per group
            group_val_results = []
            # string labels showing which tickers were in the group
            group_labels = []

            # Running training-loss accumulator within the current group
            train_loss_accum_group = 0.0

            # Iterate tickers, training sequentially
            group_tickers = []
            for ticker in tqdm(tickers_all, desc=f'Training episode {e}', ncols=100):
                L = esl.get_episode_len('train', e, ticker)
                if L < self._window_size + 1:
                    # skip tickers that don't have enough data for a window and next step
                    # Not the case for the regime episode configuration file
                    continue

                # Train on this ticker for the episode (standard rollout with replay)
                t0 = self._window_size - 1
                state = esl.get_state_matrix('train', e, ticker, t0, self._window_size)
                prev_pos = 0

                for t in range(t0, L - 1):
                    # Find action based on behaviour policy
                    action = self.act(state)

                    curr_price = float(esl.get_state_OHLCV('train', e, ticker, t)[3])
                    next_price = float(esl.get_state_OHLCV('train', e, ticker, t + 1)[3])

                    reward, pos_t = self._compute_reward(prev_pos, action, curr_price, next_price)

                    next_state = esl.get_state_matrix('train', e, ticker, t + 1, self._window_size)
                    done = (t == L - 2)

                    # store transition
                    self._memory.append((state, action, reward, next_state, done))

                    # train from replay if enough samples; accumulate training loss for this group
                    if len(self._memory) >= config['batch_size']:
                        loss = self.exp_replay(config['batch_size'])
                        train_loss_accum_group += loss

                    # advance
                    state = next_state
                    prev_pos = pos_t

                # add ticker to current group
                group_tickers.append(ticker)

                # If we've finished a group of tickers (val_group_size), run group validation
                if len(group_tickers) == config['val_group_size']:
                    # Validation on the same episode, only over these group tickers
                    val_stats = self._run_validation_group(esl, e, group_tickers, split='validate')

                    # Record
                    group_train_losses.append(train_loss_accum_group)
                    group_val_results.append(val_stats)
                    group_labels.append([",".join(group_tickers)])

                    # Reset for next group
                    train_loss_accum_group = 0.0
                    group_tickers = []

            # If there are leftover tickers in the last partial group, validate them too
            if len(group_tickers) > 0:
                val_stats = self._run_validation_group(esl, e, group_tickers, split='validate')
                group_train_losses.append(train_loss_accum_group)
                group_val_results.append(val_stats)
                group_labels.append([",".join(group_tickers)])
                train_loss_accum_group = 0.0
                group_tickers = []

            # End of episode: plot one figure (training vs validation loss over groups)
            fig_path = os.path.join(config['plots_dir'], f"ep{e}_group_losses.png")
            group_val_losses = [-d['sum_reward'] for d in group_val_results]
            self._plot_group_losses(group_train_losses, group_val_losses, fig_path)

            # Save model checkpoint
            self._model.save(os.path.join(config['model_dir'], f"model_ep{e}.keras"))

            # Save logs for this episode
            logs_by_episode[e] = {
                "group_train_losses": group_train_losses,
                "group_val_results": group_val_results,
                "group_labels": group_labels,
                "figure": fig_path,
            }

        return logs_by_episode


    def _run_validation_group(self, esl: EpisodeStateLoader, episode_id: int, tickers: list[str], split: str = 'validate'):
        # switch to test mode for validation
        prev_test_mode = self._test_mode
        self._test_mode = True

        total_reward = 0.0
        total_trades = 0
        wins = 0
        losses = 0

        for ticker in tickers:
            L = esl.get_episode_len(split, episode_id, ticker)
            if L < self._window_size + 1:
                continue

            t0 = self._window_size - 1
            state = esl.get_state_matrix(split, episode_id, ticker, t0, self._window_size)
            prev_pos = 0
            entry_price = None

            for t in range(t0, L - 1):
                action = self.act(state)
                curr_price   = float(esl.get_state_OHLCV(split, episode_id, ticker, t)[3])
                next_price = float(esl.get_state_OHLCV(split, episode_id, ticker, t + 1)[3])

                r_t, pos_t = self._compute_reward(prev_pos, action, curr_price, next_price)
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

                next_state = esl.get_state_matrix(split, episode_id, ticker, t + 1, self._window_size)
                state, prev_pos = next_state, pos_t

        self._test_mode = prev_test_mode

        hit_rate = wins / max(wins + losses, 1)
        return {
            "episode": episode_id,
            "split": split,
            "tickers": tickers,
            "sum_reward": float(total_reward),
            "total_trades": int(total_trades),
            "hit_rate": float(hit_rate),
        }

    def _plot_group_losses(self, train_losses: list[float], val_losses: list[float], fname: str | None = None, show: bool = True):

        x = np.arange(1, len(train_losses) + 1)

        fig, ax1 = plt.subplots(figsize=(10, 4))

        # Left axis: training loss
        ax1.plot(x, train_losses, marker='o', linewidth=2, color='tab:blue',
                label='Train loss (sum per group)')
        ax1.set_xlabel('Ticker group index')
        ax1.set_ylabel('Train loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, linestyle='--', alpha=0.3)

        # Right axis: validation loss
        ax2 = ax1.twinx()
        ax2.plot(x, val_losses, marker='s', linewidth=2, color='tab:orange',
                label='Validation loss (−sum reward)')
        ax2.set_ylabel('Validation loss', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Combine legends
        lines, labels = [], []
        for ax in [ax1, ax2]:
            l, lab = ax.get_legend_handles_labels()
            lines.extend(l)
            labels.extend(lab)
        ax1.legend(lines, labels, loc='best')

        plt.title('Group losses over training (per episode)')
        fig.tight_layout()

        if fname:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            plt.savefig(fname, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)