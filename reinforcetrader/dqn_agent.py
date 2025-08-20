import numpy as np
import keras

from keras import Input, layers, optimizers
from collections import deque
from tqdm import tqdm

from state import EpisodeStateLoader

@keras.saving.register_keras_serializable()
# Define DQN Model Architecture
class DualBranchDQN(keras.Model):
    def __init__(self, motif_state_shape: tuple[int, int], context_state_size: int, action_size: int):
        super().__init__()

        # Motif Branch for finding candle patterns using Conv1D
        motif_input = Input(shape=motif_state_shape, name="Motif Input")
        motif_conv_layer = layers.Conv1D(32, 3, padding="same", activation="relu")(motif_input)
        motif_conv_layer = layers.Conv1D(32, 3, padding="same", activation="relu")(motif_conv_layer)
        motif_out = layers.GlobalAveragePooling1D(name="Motif Output")(motif_conv_layer)

        # Context Branch for finding regimes
        context_input = Input(shape=(context_state_size,), name="Context Input")
        context_hid_layer = layers.Dense(64, activation="relu")(context_input)
        context_out = layers.Dense(32, activation="relu", name="Context Output")(context_hid_layer)
        
        # Late fusion of both the motif and context branches
        fused_branch = layers.Concatenate(name="Late Fusion")([motif_out, context_out])
        fused_branch = layers.Dense(128, activation="relu")(fused_branch)
        fused_branch = layers.Dense(64, activation="relu")(fused_branch)
        fused_branch = layers.LayerNormalization()(fused_branch)
        Q = layers.Dense(action_size, name="Q Values")(fused_branch)

        self._model = keras.Model(inputs=[motif_input, context_input], outputs=Q, name="DualBranchDQN")
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

    def _get_states(self, state_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Check the state matrix shape is correct
        total_features = self._num_motif_feat + self._num_context_feat
        if state_matrix.shape != (self._window_size, total_features):
            raise KeyError(f'Invalid state matrix shape: {state_matrix.shape}')
        
        # Define the motif input
        # Expand dims from [window, num_features] to [batch_size=1, window, num_features]
        motif_input = np.expand_dims(state_matrix[:, :self._num_motif_feat], axis=0).astype(np.float32)

        # Flatted the context input to [1, window * num_context_features]
        context_input = state_matrix[:, self._num_motif_feat:].reshape(1, -1).astype(np.float32)

        return motif_input, context_input
    
    def get_q_values(self, state_matrix: np.ndarray) -> np.ndarray:
        # Predict q values through DQN
        motif_input, context_input = self._get_states(state_matrix)
        q_values = self._model.predict(x=[motif_input, context_input], verbose=0)

        return q_values
    
    def fit_model(self, state_matrix: np.ndarray, target_q: np.ndarray):
        # Compute MSE loss between actual and target q values
        motif_input, context_input = self._get_states(state_matrix)
        loss = self._model.fit([motif_input, context_input], target_q, epochs=1, verbose=0)    

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
    
    
    def train(self, esl: EpisodeStateLoader, episodes_ids: list[int], batch_size: int):
        batch_losses = []
        tickers = esl.get_all_tickers()

        for e in episodes_ids:
            total_profit = 0.0
            winners = 0
            losers = 0

            for ticker in tickers:
                L = esl.get_episode_len('train', e, ticker)
                if L < self._window_size + 1:
                    # not enough ticker data for a window and a next step
                    continue 

                # initial state at t = window_size - 1
                t0 = self._window_size - 1
                state = esl.get_state_matrix('train', e, ticker, t0, self._window_size)

                prev_pos = 0  # start flat
                entry_price = None  # optional, for realized PnL logging

                # iterate until L-2 so t+1 exists
                for t in tqdm(range(t0, L - 1), desc=f'Episode {e} | {ticker}', leave=False):
                    action = self.act(state)

                    # prices for reward
                    curr_price = float(esl.get_state_OHLCV('train', e, ticker, t)[3])
                    next_price = float(esl.get_state_OHLCV('train', e, ticker, t + 1)[3])

                    # compute reward and update position
                    reward, pos_t = self._compute_reward(prev_pos, action, curr_price, next_price)

                    # optional bookkeeping for realized PnL stats when a trade ends
                    if prev_pos == 0 and pos_t == 1:
                        entry_price = curr_price  # entered at t close, pay cost inside reward
                    if prev_pos == 1 and pos_t == 0 and entry_price is not None:
                        trade_pnl = curr_price - entry_price  # realized at sell time t
                        total_profit += trade_pnl
                        if trade_pnl >= 0:
                            winners += trade_pnl
                        else:
                            losers += trade_pnl
                        entry_price = None

                    # next state
                    next_state = esl.get_state_matrix('train', e, ticker, t + 1, self._window_size)
                    done = (t == L - 2)  # last transition uses p_{L-1} -> p_{L}

                    # store transition
                    self._memory.append((state, action, reward, next_state, done))

                    # train from replay if enough samples
                    if len(self._memory) >= batch_size:
                        losses = self.exp_replay(batch_size)
                        if len(losses) > 0:
                            batch_losses.append(float(np.sum(losses)))

                    # advance
                    state = next_state
                    prev_pos = pos_t

            # save every episode if not testing
            if not self._test_mode and (e % 2 == 0):
                self._model.save(f'model_ep{e}.keras')

        return batch_losses