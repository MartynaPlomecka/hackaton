# here is mtcs_kg.py
# 
# 
from typing import Dict, Optional, List
import math
from .core import ModelState
from .knowledge_graph import CognitiveKnowledgeGraph
import logging
import numpy as np

logger = logging.getLogger(__name__)

class MCTSNode:
    def __init__(self, state: ModelState, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = [
            "add_learning_rate",
            "add_forgetting",
            "add_temperature",
            "add_bias"
        ]

class EnhancedMCTS:
    def __init__(self, knowledge_graph: CognitiveKnowledgeGraph, exploration_constant: float = 1.414):
        """
        Args:
            knowledge_graph: The cognitive knowledge graph instance.
            exploration_constant: Exploration constant for UCT.
        """
        self.kg = knowledge_graph
        self.c = exploration_constant  # exploration constant

    @property
    def exploration_constant(self):
        """Expose the exploration constant for external access."""
        return self.c

    def select_node(self, node: MCTSNode) -> MCTSNode:
        """Select node using knowledge-guided UCT."""
        visited_nodes = set()
        while node.children and not node.untried_actions:
            if node in visited_nodes:
                logger.warning("Cycle detected in MCTS tree. Breaking loop.")
                break
            visited_nodes.add(node)
            node = self._select_knowledge_uct(node)
        return node


    def expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Expand node using knowledge-guided action selection."""
        if not node.untried_actions:
            return None

        action = self._select_action_with_knowledge(node)
        node.untried_actions.remove(action)

        new_state = self._apply_action(node.state, action)
        child = MCTSNode(new_state, parent=node)
        node.children.append(child)
        return child

    def _select_knowledge_uct(self, node: MCTSNode) -> MCTSNode:
        """UCT selection with knowledge graph guidance."""
        def uct_value(child: MCTSNode) -> float:
            if child.visits == 0:
                return float('inf')

            exploitation = child.value / child.visits
            exploration = self.c * math.sqrt(math.log(node.visits) / child.visits)
            kg_prior = self._compute_kg_prior(child.state)

            return exploitation + exploration + kg_prior

        return max(node.children, key=uct_value)

    def _compute_kg_prior(self, state: ModelState) -> float:
        """Compute prior based on knowledge graph."""
        try:
            mechanism = self._extract_mechanism(state)
            info = self.kg.query_mechanism(mechanism)

            similarity_score = self._compute_model_similarity(
                state, info.get('best_model', None)
            )
            mechanism_score = info.get('best_score', 0.0)

            return 0.3 * similarity_score + 0.2 * mechanism_score

        except Exception as e:
            logger.error(f"Error computing KG prior: {e}")
            return 0.0

    def _select_action_with_knowledge(self, node: MCTSNode) -> str:
        """Select action using knowledge graph guidance."""
        mechanism = self._extract_mechanism(node.state)
        info = self.kg.query_mechanism(mechanism)

        action_scores = {}
        for action in node.untried_actions:
            score = self._compute_action_score(action, info)
            action_scores[action] = score

        total_score = sum(action_scores.values())
        if total_score == 0:
            return np.random.choice(node.untried_actions)

        probs = [score / total_score for score in action_scores.values()]
        return np.random.choice(list(action_scores.keys()), p=probs)

    def _compute_action_score(self, action: str, mechanism_info: Dict) -> float:
        """Compute score for an action based on mechanism knowledge."""
        if not mechanism_info:
            return 1.0

        if 'parameters' in mechanism_info:
            if action == "add_learning_rate" and 'learning_rate' in mechanism_info['parameters']:
                return 2.0
            if action == "add_temperature" and 'temperature' in mechanism_info['parameters']:
                return 1.5

        return 1.0

    def _apply_action(self, state: ModelState, action: str) -> ModelState:
        """Apply an action to create a new state."""
        new_state = state.copy()

        actions = {
            "add_learning_rate": {
                "param": "alpha",
                "value": 0.1,
                "equation": lambda eq: f"({eq}) * alpha"
            },
            "add_forgetting": {
                "param": "gamma",
                "value": 0.1,
                "equation": lambda eq: f"({eq}) * (1 - gamma)"
            },
            "add_temperature": {
                "param": "temp",
                "value": 1.0,
                "equation": lambda eq: f"({eq}) / temp"
            },
            "add_bias": {
                "param": "bias",
                "value": 0.0,
                "equation": lambda eq: f"({eq}) + bias"
            }
        }

        if action in actions:
            action_info = actions[action]
            new_state.equations = [
                action_info["equation"](new_state.equations[0])
            ]
            new_state.parameters[action_info["param"]] = action_info["value"]

        return new_state

    def _extract_mechanism(self, state: ModelState) -> str:
        """Extract mechanism type from model."""
        if "Q(t)" in state.equations[0]:
            return "reinforcement_learning"
        if "WM(t)" in state.equations[0]:
            return "working_memory"
        return "unknown_mechanism"

    def _compute_model_similarity(self, model: ModelState, reference: Optional[Dict]) -> float:
        """Compute similarity between model and reference."""
        if not reference:
            return 0.0

        try:
            model_eq = model.equations[0]
            ref_eq = reference['equations'][0]

            model_terms = set(model_eq.split())
            ref_terms = set(ref_eq.split())

            similarity = len(model_terms.intersection(ref_terms)) / len(model_terms.union(ref_terms))
            return similarity

        except Exception as e:
            logger.error(f"Error computing model similarity: {e}")
            return 0.0
