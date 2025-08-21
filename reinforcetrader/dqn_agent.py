import os
import numpy as np
import matplotlib.pyplot as plt

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
    
    def get_model(self):
        return self._model

    def _init_model(self) -> keras.Model:
        # Compute state shapes for the model
        motif_shape = (self._window_size, self._num_motif_feat)
        context_size = self._window_size * self._num_context_feat

        # Init model
        dual_dqn = DualBranchDQN(motif_shape, context_size, self._action_size)

        return dual_dqn.get_model()

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

        # take the last batch_size items
        mini_batch = [self._memory[i] for i in range(len(self._memory) - batch_size, len(self._memory))]
        losses = []

        for state, action, reward, next_state, done in mini_batch:
            # current Q
            q_current = self.get_q_values(state)  # shape [1, A]

            if done or next_state is None:
                target_for_action = reward
            else:
                q_next = self.get_q_values(next_state)
                target_for_action = reward + self._gamma * float(np.max(q_next, axis=1))

            # set target
            q_target = q_current.copy()
            q_target[0, action] = target_for_action

            hist = self.fit_model(state, q_target)
            losses.extend(hist.history.get('loss', []))

        # epsilon decay
        if self._epsilon > self._epsilon_min and not self._test_mode:
            self._epsilon *= self._epsilon_decay

        return losses
    
    def _compute_reward(self, prev_pos: int, action: int, curr_price: float, next_price: float) -> tuple[float, int]:
        # Update position from action
        if action == 0:
            pos_t = prev_pos
        elif action == 1:
            pos_t = 1
        elif action == 2:
            pos_t = 0

        ret = (next_price - curr_price) / max(curr_price, 1e-12)
        trade_cost = 0.0005 if pos_t != prev_pos else 0.0
        reward = pos_t * ret - trade_cost
        return reward, pos_t
    
    def train(self, esl: EpisodeStateLoader, episodes_ids: list[int], batch_size: int, val_group_size: int = 5, out_dir: str='runs/'):
        # Make runs directory if it does not exist
        os.makedirs(out_dir, exist_ok=True)

        logs_by_episode = {}

        tickers_all = esl.get_all_tickers()

        for e in episodes_ids:
            # Per-episode per-group stores of training performance
            group_train_losses = []   # list of floats, one per group
            group_val_losses   = []   # list of floats, one per group
            group_labels       = []   # string labels showing which tickers were in the group

            # Running training-loss accumulator within the current group
            train_loss_accum_group = 0.0

            # Iterate tickers, training sequentially
            group_tickers = []
            for ticker in tickers_all:
                L = esl.get_episode_len('train', e, ticker)
                if L < self._window_size + 1:
                    # skip tickers that don't have enough data for a window and next step
                    continue

                # Train on this ticker for the episode (standard rollout with replay) ----
                t0 = self._window_size - 1
                state = esl.get_state_matrix('train', e, ticker, t0, self._window_size)
                prev_pos = 0
                entry_price = None

                for t in tqdm(range(t0, L - 1), desc=f'Train ep {e} | {ticker}', ncols=100):
                    action = self.act(state)

                    curr_price = float(esl.get_state_OHLCV('train', e, ticker, t)[3])
                    next_price = float(esl.get_state_OHLCV('train', e, ticker, t + 1)[3])

                    reward, pos_t = self._compute_reward(prev_pos, action, curr_price, next_price)

                    # optional realized pnl bookkeeping (not used for losses here)
                    if prev_pos == 0 and pos_t == 1:
                        entry_price = curr_price
                    if prev_pos == 1 and pos_t == 0 and entry_price is not None:
                        # realized at sell; not needed for loss
                        entry_price = None

                    next_state = esl.get_state_matrix('train', e, ticker, t + 1, self._window_size)
                    done = (t == L - 2)

                    # store transition
                    self._memory.append((state, action, reward, next_state, done))

                    # train from replay if enough samples; accumulate training loss for this group
                    if len(self._memory) >= batch_size:
                        losses = self.exp_replay(batch_size)
                        if len(losses) > 0:
                            train_loss_accum_group += float(np.sum(losses))

                    # advance
                    state = next_state
                    prev_pos = pos_t

                # add ticker to current group
                group_tickers.append(ticker)

                # If we've finished a group of tickers (val_group_size), run group validation
                if len(group_tickers) == val_group_size:
                    # Validation on the same episode, only over these group tickers
                    val_stats = self._run_validation_group(esl, e, group_tickers, split='validate')

                    # Define validation "loss" as negative sum_reward for comparability with training loss
                    val_loss = -float(val_stats["sum_reward"])

                    # Record
                    group_train_losses.append(train_loss_accum_group)
                    group_val_losses.append(val_loss)
                    group_labels.append(",".join(group_tickers))

                    # Reset for next group
                    train_loss_accum_group = 0.0
                    group_tickers = []

            # If there are leftover tickers in the last partial group, validate them too
            if len(group_tickers) > 0:
                val_stats = self._run_validation_group(esl, e, group_tickers, split='validate')
                val_loss = -float(val_stats["sum_reward"])
                group_train_losses.append(train_loss_accum_group)
                group_val_losses.append(val_loss)
                group_labels.append(",".join(group_tickers))
                train_loss_accum_group = 0.0
                group_tickers = []

            # End of episode: plot one figure (training vs validation loss over groups) ----
            fig_path = os.path.join(out_dir, f"ep{e}_group_losses.png")
            self._plot_group_losses(group_train_losses, group_val_losses, fig_path)

            # Save model checkpoint
            if not self._test_mode:
                self._model.save(os.path.join(out_dir, f"model_ep{e}.keras"))

            # Save logs for this episode
            logs_by_episode[e] = {
                "group_train_losses": group_train_losses,
                "group_val_losses": group_val_losses,
                "group_labels": group_labels,
                "figure": fig_path,
            }

        return logs_by_episode


    def _run_validation_group(self, esl: EpisodeStateLoader, episode_id: int, tickers: list[str], split: str = 'validate'):
        eps_backup = self._epsilon
        self._epsilon = 0.0  # greedy

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
                action = self.act(state)  # greedy due to epsilon=0
                p_t   = float(esl.get_state_OHLCV(split, episode_id, ticker, t)[3])
                p_tp1 = float(esl.get_state_OHLCV(split, episode_id, ticker, t + 1)[3])

                r_t, pos_t = self._compute_reward(prev_pos, action, p_t, p_tp1)
                total_reward += r_t

                # simple trade bookkeeping (optional)
                if prev_pos == 0 and pos_t == 1:
                    entry_price = p_t
                    total_trades += 1
                if prev_pos == 1 and pos_t == 0 and entry_price is not None:
                    trade_pnl = p_t - entry_price
                    if trade_pnl >= 0:
                        wins += 1
                    else:
                        losses += 1
                    entry_price = None

                next_state = esl.get_state_matrix(split, episode_id, ticker, t + 1, self._window_size)
                state, prev_pos = next_state, pos_t

        self._epsilon = eps_backup

        hit_rate = wins / max(wins + losses, 1)
        return {
            "episode": episode_id,
            "split": split,
            "tickers": tickers,
            "sum_reward": float(total_reward),
            "total_trades": int(total_trades),
            "hit_rate": float(hit_rate),
        }


    def _plot_group_losses(self, train_losses: list[float], val_losses: list[float], fname: str | None = None):
        x = np.arange(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 4))