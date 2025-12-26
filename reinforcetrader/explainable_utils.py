import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from dqn_agent import DualBranchDQN, DRLAgent

class ModelExplainer:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        
        try:
            # Load the pre-trained model
            self._model = keras.models.load_model(model_path)
            print(f"Model successfully loaded from {model_path}")
        except FileNotFoundError:
            raise ValueError(f"Error: The model file was not found at {model_path}")

    def get_model_summary(self) -> None:
        self._model.summary()
    
    def get_model_input(self, state_matrix: pd.DataFrame, trade_pos, rew_pars: dict[str, float]):
        # Convert the model inputs to their respective data types
        state_input = np.expand_dims(state_matrix.to_numpy(), axis=0).astype(np.float32)
        complete_reward_params = [trade_pos] + list(rew_pars.values())
        reward_input = np.array([complete_reward_params], dtype=np.float32)
        
        # Return the inputs as dict
        return {'state_input': state_input, 'reward_input': reward_input}
    
    def get_1d_grad_cam(self, state: pd.DataFrame, rew_pars: dict[str, float], trade_pos, action: int, layer_name: str):
        # Check if trade position and action are valid
        if trade_pos not in {DRLAgent.IN_TRADE, DRLAgent.OUT_TRADE}:
            raise ValueError(f"Trade position must be either {DRLAgent.IN_TRADE} or {DRLAgent.OUT_TRADE}.")
        if action not in {DRLAgent.A_BUY, DRLAgent.A_SELL, DRLAgent.A_HOLD}:
            raise ValueError(f"Action must be supported by the DRL Agent.")
        
        # Validate layer name and input shapes
        if layer_name not in [layer.name for layer in self._model.layers]:
            raise ValueError(f"Layer name {layer_name} not found in the model.")
        if state.shape != (self._model.input_shape[0][1], self._model.input_shape[0][2]):
            raise ValueError(f"{state.shape} does not match state input shape {self._model.input_shape[0][1:]}.")
        if len(rew_pars) + 1 != self._model.input_shape[1][1]:
            raise ValueError(f"Reward parameters length must be {len(rew_pars) + 1}")

        # Get the model inputs for feed-forward
        model_inputs = self.get_model_input(state, trade_pos, rew_pars)
        
        # Create a model that outputs the layer output and the final prediction
        grad_model = keras.Model(
            inputs = self._model.inputs,
            outputs = [self._model.get_layer(layer_name).output, self._model.output]
        )
        
        # Record operations so that we can differentiate the output w.r.t input layer
        with tf.GradientTape() as tape:
            # feed forward
            conv_outputs, q_values = grad_model(model_inputs)
            # Get q_value for the selected action
            # NOTE: Action const. labels coincides with q_value index
            loss = q_values[:, action]
        
        # Compute d(q_a)/d(layer_output)
        grads = tape.gradient(loss, conv_outputs)
        
        # Computes the weights for CAM, i.e., for sum_k (w_k * A_k)
        weights = tf.reduce_mean(grads, axis=(0, 1))
        
        # Compute the weighted combination of activations and weights
        conv_outputs = conv_outputs[0] # Remove batch dimension
        # NOTE: conv_outputs is 60x32 and weights is 32x1. Result is 60x1
        heatmap = conv_outputs @ weights[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU to the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
        