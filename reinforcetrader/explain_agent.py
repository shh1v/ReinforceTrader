import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
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
    
    def __init__(self, model_path: str, reward_function: str) -> None:
        self._model_path = model_path
        
        try:
            # Load the pre-trained model
            self._model = keras.models.load_model(model_path)
            print(f"Model successfully loaded from {model_path}")
        except FileNotFoundError:
            raise ValueError(f"Error: The model file was not found at {model_path}")
        
        # Store the state and rewardfeature names
        self._state_feat_names = list(self.DBDQN_FEATURE_PARAMS.keys())
        reward_function_params = DRLAgent.reward_param_keys(reward_function)
        self._rew_feat_names = ['trade_pos'] + reward_function_params[1:] # Exclude Rt
        
        # Set flags for SHAP explainer
        self._shap_baseline_set = False

    def get_model_summary(self) -> None:
        self._model.summary()
    
    def get_model_input(self, state_matrix: np.ndarray, reward_params: np.ndarray) -> dict[str, np.ndarray]:
        # Convert the model inputs to their respective data types
        state_input = np.expand_dims(state_matrix, axis=0).astype(np.float32)
        reward_input = np.array([reward_params], dtype=np.float32)
        
        # Return the inputs as dict
        return {'state_input': state_input, 'reward_input': reward_input}
    
    def _1d_grad_cam_heatmap(self, state: np.ndarray, rew_pars: np.ndarray, action: int, layer_name: str):
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
        if len(rew_pars) != rew_pars_len:
            raise ValueError(f"Reward parameters length must be {len(rew_pars)}")

        # Get the model inputs for feed-forward
        model_inputs = self.get_model_input(state, rew_pars)
        
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

    def run_grad_cam(self, state: np.ndarray, rew_pars: np.ndarray, action: int, layer_name: str) -> None:
            # Generate the heatmap (Shape: (window,))
            M = self._1d_grad_cam_heatmap(state, rew_pars, action, layer_name)
            # Reverse the M so that the largest index has the value for most recent time step
            rev_M = M[::-1]
            
            # Prepare the state matrix for plotting
            # Transpose and reverse. So, y axis is features, x has time steps in chronological order
            vis_state = state.T[:, ::-1]
            
            # Get the state image based on feature range and values
            state_image = self._build_state_image(vis_state)
            
            # Setup the subplots, with shared x-axis
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot the state image
            ax1.imshow(state_image, aspect='auto', interpolation='nearest')
            ax1.set_yticks(np.arange(len(self._state_feat_names)))
            ax1.set_yticklabels(self._state_feat_names, fontsize=8, fontweight='bold')
            ax1.set_title('State Features Over Time', fontsize=12, fontweight='bold')
            ax1.set_yticks(np.arange(len(self._state_feat_names)) - 0.5, minor=True)
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
            
    def _build_state_image(self, vis_state: np.ndarray) -> np.ndarray:
        # Prepare the matrix to store the color values
        image_matrix = np.zeros((vis_state.shape[0], vis_state.shape[1], 4), dtype=np.float32)
        
        # Define Base Colors to compute the color values
        COLORS = {
            'red':   np.array([0.8, 0.0, 0.0]), # Red
            'green': np.array([0.0, 0.5, 0.0]), # Green
            'pink':  np.array([0.6, 0.0, 0.6]), # Purple/Magenta
            'yellow': np.array([0.8, 0.8, 0.0]) # Yellow
        }
        
        for i, feature_name in enumerate(self._state_feat_names):
            # Get feature values and configuration
            feat_vals = vis_state[i]
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
    
    def setup_shap_explainer(self, bg_states: np.ndarray, bg_rewards: np.ndarray) -> None:
        if self._shap_baseline_set:
            print("SHAP explainer already setup. Skipping..")
            return
        
        # Get the input shapes for keras model
        state_input_shape = self._model.get_layer('state_input').output.shape
        reward_input_shape = self._model.get_layer('reward_input').output.shape
        
        # Create wrapper model input
        w_state_input = keras.Input(shape=state_input_shape[1:], name='w_state_input')
        w_reward_input = keras.Input(shape=reward_input_shape[1:], name='w_reward_input')
        w_output = self._model({'state_input': w_state_input, 'reward_input': w_reward_input}, training=False)
        
        # Create the wrapper model
        w_model = keras.Model(inputs=[w_state_input, w_reward_input], outputs=w_output)
        
        # Create a SHAP explainer using DeepExplainer (requires model input as list)
        self._shap_explainer = shap.GradientExplainer(w_model, [bg_states, bg_rewards])
        
        # SHAP baseline is now set
        self._shap_baseline_set = True
    
    def run_shap(self, state: np.ndarray, rew_pars: np.ndarray, action: int) -> None:
        # Check if action is valid for the trade_pos
        if rew_pars[0] == DRLAgent.OUT_TRADE and action == DRLAgent.A_SELL:
            raise ValueError("Cannot explain SELL action when not in trade.")
        if rew_pars[0] == DRLAgent.IN_TRADE and action == DRLAgent.A_BUY:
            raise ValueError("Cannot explain BUY action when already in trade.")
        
        # Check if SHAP explainer is setup
        if not self._shap_baseline_set:
            raise ValueError("SHAP explainer not setup. Call setup_shap_explainer() first.")
        
        # State and reward inputs don't have batch dimension
        # So expand dims, and create model input as list
        state_input = np.expand_dims(state, axis=0).astype(np.float32)
        reward_input = np.array([rew_pars], dtype=np.float32)
        model_input = [state_input, reward_input]
        
        # Compute the SHAP values for the target input
        # Rseed set for reproducibility (see https://github.com/shap/shap/issues/1010)
        shap_values = self._shap_explainer.shap_values(model_input, rseed=42)
        
        # Separate SHAP values for state and reward parameters
        state_shap_vals = shap_values[0][0, :, :, action] # type: ignore
        reward_shap_vals = shap_values[1][0, :, action] # type: ignore
        
        # Plot the SHAP state and reward values as heatmap
        self._plot_shap_values(state_shap_vals, reward_shap_vals, reward_input, action)
        
    def _plot_shap_values(self, state_shap, reward_shap, reward_raw, action):
        # Transpose and reverse the state SHAP for visualization
        vis_shap = state_shap.T[:, ::-1]
        
        # Setup Figure: Top = Heatmap, Bottom = Bar Chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                       gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})
        
        # First, plot the SHAP heatmap for state features
        heat_val_lim = np.max(np.abs(vis_shap))
        im = ax1.imshow(vis_shap, aspect='auto', cmap='seismic', vmin=-heat_val_lim, vmax=heat_val_lim)
        
        # Formatting Y-Axis
        ax1.set_yticks(np.arange(len(self._state_feat_names)))
        ax1.set_yticklabels(self._state_feat_names, fontsize=9, fontweight='bold')
        act_str = {DRLAgent.A_BUY: "Buy", DRLAgent.A_SELL: "Sell", DRLAgent.A_HOLD: "Hold"}.get(action, "N/A")
        ax1.set_title(f"State SHAP Values (Red = Pushes for {act_str} action, Blue = Against)", fontweight='bold', fontsize=12)
        
        # Formatting X-Axis (Time)
        n_timesteps = vis_shap.shape[1]
        ticks = np.arange(0, n_timesteps, 5)
        labels = [f"t={n_timesteps - 1 - t}" for t in ticks]
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labels)
        ax1.set_xlabel("Time Lag (Right = Latest)")
        
        # Colorbar and grid lines
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('SHAP Impact', rotation=270, labelpad=15)
        
        ax1.set_yticks(np.arange(len(self._state_feat_names)) - 0.5, minor=True)
        ax1.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.1)
        ax1.tick_params(which="minor", bottom=False, left=False)

        # Second, plot the SHAP bar chart for reward parameters
        colors = ['#d62728' if x > 0 else '#1f77b4' for x in reward_shap] # Red/Blue
        y_pos = np.arange(len(self._rew_feat_names))
        
        ax2.barh(y_pos, reward_shap, color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(self._rew_feat_names, fontweight='bold', fontsize=10)
        
        ax2.axvline(0, color='black', linewidth=0.8)
        ax2.set_xlabel(f"SHAP Value (Impact on Q-Value)")
        ax2.set_title(f"Impact of Reward Parameters & Position", fontweight='bold', fontsize=12)
        ax2.grid(axis='x', alpha=0.3)

        # Annotate bars with values
        for i, v in enumerate(reward_shap):
            feat_val = reward_raw[0][i]
            ax2.text(v, i, f" Val: {feat_val:.2e}", va='center', fontsize=9)

        plt.show()