import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from .dqn_agent import DRLAgent

class ModelExplainer:
    DBDQN_FEATURE_PARAMS = {
        'Body/HL': {'vmin': -1.0, 'vmax': 1.0, 'color':'redgreen'},
        'UWick/HL': {'vmin': 0.0, 'vmax': 1.0, 'color':'red'},
        'LWick/HL': {'vmin': 0.0, 'vmax': 1.0, 'color':'green'},
        'Gap': {'vmin': -3.0, 'vmax': 3.0, 'color':'redgreen'},
        'GapFill': {'vmin': 0.0, 'vmax': 1.0, 'color':'pink'},
        'EMA5/13': {'vmin': -3.0, 'vmax': 3.0, 'color':'redgreen'},
        'EMA13/26': {'vmin': -3.0, 'vmax': 3.0, 'color':'redgreen'},
        'EMA26/50': {'vmin': -3.0, 'vmax': 3.0, 'color':'redgreen'},
        'B%B': {'vmin': 0.0, 'vmax': 1.0, 'color':'redgreen'},
        'BBW': {'vmin': -3.0, 'vmax': 3.0, 'color':'pink'},
        'RSI': {'vmin': -3.0, 'vmax': 3.0, 'color':'redgreen'},
        'ADX': {'vmin': 0.0, 'vmax': 1.0, 'color':'pink'},
        'V/Vol20': {'vmin': -3.0, 'vmax': 3.0, 'color':'redgreen'},
    }
    
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
    
    def _1d_grad_cam_heatmap(self, state: pd.DataFrame, rew_pars: dict[str, float], trade_pos, action: int, layer_name: str):
        # Check if trade position and action are valid
        if trade_pos not in {DRLAgent.IN_TRADE, DRLAgent.OUT_TRADE}:
            raise ValueError(f"Trade position must be either {DRLAgent.IN_TRADE} or {DRLAgent.OUT_TRADE}.")
        if action not in {DRLAgent.A_BUY, DRLAgent.A_SELL, DRLAgent.A_HOLD}:
            raise ValueError(f"Action must be supported by the DRL Agent.")
        
        # Check whether layer name exists
        if layer_name not in [layer.name for layer in self._model.layers]:
            raise ValueError(f"Layer name {layer_name} not found in the model.")
        
        # Validate input shapes
        state_input_shape = self._model.get_layer('state_input').output.shape
        rew_pars_len = self._model.get_layer('reward_input').output.shape[1]
        if state.shape != state_input_shape[1:]:
            raise ValueError(f"{state.shape} does not match state input shape {self._model.input_shape[0][1:]}.")
        if len(rew_pars) + 1 != rew_pars_len:
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
        M = conv_outputs @ weights[..., tf.newaxis]
        M = tf.squeeze(M)
        
        # Apply ReLU to the weighted sum
        M = tf.maximum(M, 0) / tf.math.reduce_max(M)
        
        return M.numpy()

    def run_grad_cam(self, state: pd.DataFrame, rew_pars: dict[str, float], trade_pos, action: int, layer_name: str) -> None:
            # Generate the heatmap (Shape: (window,))
            M = self._1d_grad_cam_heatmap(state, rew_pars, trade_pos, action, layer_name)
            # Reverse the M so that the largest index has the value for most recent time step
            rev_M = M[::-1]
            
            # Prepare the state matrix for plotting
            # Transpose and reverse. So, y axis is features, x has time steps in chronological order
            vis_state = state.T.iloc[:, ::-1]
            
            # Get the state image based on feature range and values
            state_image = self._build_state_image(vis_state)
            
            # Setup the subplots, with shared x-axis
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot the state image
            ax1.imshow(state_image, aspect='auto', interpolation='nearest')
            ax1.set_yticks(np.arange(len(vis_state.index)))
            ax1.set_yticklabels(vis_state.index, fontsize=8, fontweight='bold')
            ax1.set_title('State Features Over Time', fontsize=12, fontweight='bold')
            ax1.set_yticks(np.arange(len(vis_state.index)) - 0.5, minor=True)
            ax1.grid(which="minor", color="grey", linestyle='-', linewidth=0.5, alpha=0.3)
            ax1.tick_params(which="minor", bottom=False, left=False)
            
            # Plot the Grad-CAM importance scores (heatmap)
            time_indices = np.arange(len(rev_M))
            ax2.bar(time_indices, rev_M, color='orange', width=1.0, alpha=0.8)
            
            ax2.set_title('Grad-CAM Importance Scores', fontsize=12, fontweight='bold')
            ax2.set_ylabel("Scores [0, 1]")
            ax2.set_ylim(0, 1.1)
            ax2.grid(True, axis='y', alpha=0.3)

            ticks = np.arange(0, len(rev_M), 5)
            max_lag = len(rev_M) - 1
            labels = [f"t={max_lag - t}" for t in ticks]
            
            ax2.set_xticks(ticks)
            ax2.set_xticklabels(labels)
            ax2.set_xlabel("Time Lag (Right = Latest)")

            plt.subplots_adjust(right=0.85)
            plt.show()
            
    def _build_state_image(self, vis_state: pd.DataFrame) -> np.ndarray:
        # Prepare the matrix to store the color values
        image_matrix = np.zeros((vis_state.shape[0], vis_state.shape[1], 4), dtype=np.float32)
        
        # Define Base Colors to compute the color values
        COLORS = {
            'red':   np.array([0.8, 0.0, 0.0]), # Deep Red
            'green': np.array([0.0, 0.5, 0.0]), # Deep Green
            'pink':  np.array([0.6, 0.0, 0.6]), # Deep Purple/Magenta
            'yellow': np.array([0.8, 0.8, 0.0]) # Yellow
        }
        
        for i, feature_name in enumerate(vis_state.index):
            # Get feature values and configuration
            feat_vals = vis_state.iloc[i].to_numpy()
            feat_config = self.DBDQN_FEATURE_PARAMS.get(feature_name, None)
            if feat_config is None:
                raise ValueError(f"Feature {feature_name} not found in DBDQN_FEATURE_PARAMS.")
            vmin, vmax, color = feat_config['vmin'], feat_config['vmax'], feat_config['color']
            
            if vmin >= vmax:
                raise ValueError(f"Invalid vmin and vmax for feature {feature_name}.")
            
            if color == 'redgreen':
                # Use red for vals<mid, green for vals>=mid
                mid = (vmax + vmin) / 2
                ltm_vals = feat_vals < mid
                gtem_vals = ~ ltm_vals
                
                # Apply color mapping for negative values
                if np.any(ltm_vals):
                    ltm_range = mid - vmin
                    val_dist = mid - feat_vals[ltm_vals]
                    
                    alpha_vals = np.clip(val_dist / ltm_range, 0.0, 1.0)
                    
                    image_matrix[i, ltm_vals, :3] = COLORS['red']
                    image_matrix[i, ltm_vals, 3] = alpha_vals
                if np.any(gtem_vals):
                    gtem_range = vmax - mid
                    val_dist = feat_vals[gtem_vals] - mid
                    
                    alpha_vals = np.clip(val_dist / gtem_range, 0.0, 1.0)
                    
                    image_matrix[i, gtem_vals, :3] = COLORS['green']
                    image_matrix[i, gtem_vals, 3] = alpha_vals
            else:
                # Apply single color mapping for all values
                rgb = COLORS.get(color, None)
                if rgb is None:
                    raise ValueError(f"Color {color} not defined in COLORS.")
                v_range = vmax - vmin
                val_dist = feat_vals - vmin
                
                alpha_vals = np.clip(val_dist / v_range, 0.0, 1.0)
                
                image_matrix[i, :, :3] = rgb
                image_matrix[i, :, 3] = alpha_vals
                
        return image_matrix