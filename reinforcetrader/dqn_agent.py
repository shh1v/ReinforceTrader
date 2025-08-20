import numpy as np
import keras

from keras import Input, layers, optimizers
from collections import deque

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

    def exp_replay(self, batch_size):
        losses = []
        
        # Define mini-batch which holds batch_size most recent states from memory
        mini_batch = []
        for i in range(len(self._memory) - batch_size + 1, len(self._memory)):
            mini_batch.append(self._memory[i])
            
        for state, action, reward, next_state, done in mini_batch:
            if done:
                # special condition for last training epoch in batch (no next_state)
                optimal_q_for_action = reward  
            else:
                # target Q-value is updated using the Bellman equation: reward + gamma * max(predicted Q-value of next state)
                optimal_q_for_action = reward + self._gamma * np.max(self.get_q_values(next_state))
                
            # Get the predicted Q-values of the current state
            q_table = self.get_q_values(state)
            # Update the output Q table - replace the predicted Q value for action with the target Q value for action 
            q_table[0][action] = optimal_q_for_action
            # Fit the model where state is X and target_q_table is Y
            history = self.fit_model(state, q_table)
            losses += history.history['loss']
           
        # define epsilon decay (for the act function)
        if self.epsilon > self._epsilon_min:
            self.epsilon *= self._epsilon_decay
        return losses