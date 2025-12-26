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