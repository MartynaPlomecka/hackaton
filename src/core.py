# src/core.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import uuid

@dataclass
class ModelState:
    """Represents a state in our cognitive model search"""
    equations: List[str]  # The mathematical equations of the model
    parameters: Dict[str, float]  # Model parameters (e.g., learning rate)
    score: Optional[float] = None  # How well the model performs
    visits: int = 0  # For MCTS tracking
    value: float = 0.0  # For MCTS tracking
    id: str = ''  # Unique identifier for graph operations
    
    def __post_init__(self):
        """Initialize unique ID if not provided"""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def __hash__(self):
        """Make ModelState hashable using ID"""
        return hash(self.id)
    
    def __eq__(self, other):
        """Equality comparison using ID"""
        if not isinstance(other, ModelState):
            return False
        return self.id == other.id
    
    def copy(self):
        """Create a deep copy of the state"""
        return ModelState(
            equations=self.equations.copy(),
            parameters=self.parameters.copy(),
            score=self.score,
            visits=self.visits,
            value=self.value,
            id=str(uuid.uuid4())
        )

def generate_test_data(n_trials: int = 100) -> Dict:
    """Generate synthetic two-armed bandit data"""
    np.random.seed(42)  
    
    data = {
        'timestamps': np.arange(n_trials),
        'actions': np.random.binomial(1, 0.5, n_trials),  # Random actions (0 or 1)
        'rewards': np.zeros(n_trials)
    }
    
    # Generate rewards based on action chosen
    # Action 1 has 0.7 probability of reward, Action 0 has 0.3, this is totally made up for now
    for t in range(n_trials):
        prob = 0.7 if data['actions'][t] == 1 else 0.3
        data['rewards'][t] = np.random.binomial(1, prob)
    
    return data