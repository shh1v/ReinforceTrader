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
        self._model = keras.models.load_model.load(model_name) if self._test_mode else self._init_model()
    
    def _init_model(self) -> keras.Model:
        # Compute state shapes for the model
        motif_shape = (self._window_size, self._num_motif_feat)
        context_size = self._window_size * self._num_context_feat

        # Init model
        dual_dqn = DualBranchDQN(motif_shape, context_size, self._action_size)

        return dual_dqn.get_model()