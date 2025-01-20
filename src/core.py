# src/core.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class ModelState:
    """Represents a state in our cognitive model search"""
    equations: List[str]  # The mathematical equations of the model
    parameters: Dict[str, float]  # Model parameters (e.g., learning rate)
    score: Optional[float] = None  # How well the model performs
    visits: int = 0  # For MCTS tracking
    value: float = 0.0  # For MCTS tracking
    
    def copy(self):
        """Create a deep copy of the state"""
        return ModelState(
            equations=self.equations.copy(),
            parameters=self.parameters.copy(),
            score=self.score,
            visits=self.visits,
            value=self.value
        )

def generate_test_data(n_trials: int = 100) -> Dict:
    """Generate synthetic two-armed bandit data"""
    np.random.seed(42)  
    
    # Init data structures
    data = {
        'timestamps': np.arange(n_trials),
        'actions': np.random.binomial(1, 0.5, n_trials),  # Random actions (0 or 1)
        'rewards': np.zeros(n_trials)
    }
    
    # Generate rewards based on action chosen
    # Action 1 has 0.7 probability of reward, Action 0 has 0.3, just now, needs to be fixed
    for t in range(n_trials):
        prob = 0.7 if data['actions'][t] == 1 else 0.3
        data['rewards'][t] = np.random.binomial(1, prob)
    
    return data