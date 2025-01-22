# mcts_kg.py

# - Search algorithm that explores different model variations
# - Uses knowledge graph to guide search
# - !!!! has graph-based tracking of model relationships !!!
# - Can combine and refine models based on their relationships

from typing import Dict, Optional, List
import math
import networkx as nx
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
        # GoT: Track thought relationships
        self.thought_parents: List[MCTSNode] = []
        self.refinement_count = 0

class EnhancedMCTS:
    def __init__(self, knowledge_graph: CognitiveKnowledgeGraph, exploration_constant: float = 1.414):
        """
        Args:
            knowledge_graph: The cognitive knowledge graph instance.
            exploration_constant: Exploration constant for UCT.
        """
        self.kg = knowledge_graph
        self.c = exploration_constant  # exploration constant
        # GoT: Track thought relationships in a graph
        self.thought_graph = nx.DiGraph()

    @property
    def exploration_constant(self):
        """Expose the exploration constant for external access."""
        return self.c

    def select_node(self, node: MCTSNode) -> MCTSNode:
        """Select node using knowledge-guided UCT with GoT metrics."""
        visited_nodes = set()
        while node.children and not node.untried_actions:
            if node in visited_nodes:
                logger.warning("Cycle detected in MCTS tree. Breaking loop.")
                break
            visited_nodes.add(node)
            node = self._select_knowledge_uct(node)
            
            # GoT: Track relationship in thought graph
            self._add_to_thought_graph(node)
        return node

    def _add_to_thought_graph(self, node: MCTSNode):
        """GoT: Add node and its relationships to thought graph"""
        if not self.thought_graph.has_node(node.state.id):
            self.thought_graph.add_node(node.state.id, state=node.state)
            # Add edges from parent thoughts
            for parent in node.thought_parents:
                if parent and self.thought_graph.has_node(parent.state.id):
                    self.thought_graph.add_edge(parent.state.id, node.state.id)

    def expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Expand node using knowledge-guided action selection."""
        if not node.untried_actions:
            return None

        action = self._select_action_with_knowledge(node)
        node.untried_actions.remove(action)

        new_state = self._apply_action(node.state, action)
        child = MCTSNode(new_state, parent=node)
        # GoT: Track parent relationship
        child.thought_parents.append(node)
        node.children.append(child)
        
        # GoT: Update thought graph
        self._add_to_thought_graph(child)
        
        return child

    def _select_knowledge_uct(self, node: MCTSNode) -> MCTSNode:
        """UCT selection with knowledge graph guidance and GoT metrics."""
        def uct_value(child: MCTSNode) -> float:
            if child.visits == 0:
                return float('inf')

            exploitation = child.value / child.visits
            exploration = self.c * math.sqrt(math.log(node.visits) / child.visits)
            kg_prior = self._compute_kg_prior(child.state)
            
            # GoT: Add volume bonus and latency penalty
            volume_score = self._compute_volume_score(child.state)
            latency_penalty = self._compute_latency_penalty(child.state)

            return exploitation + exploration + kg_prior + volume_score - latency_penalty

        return max(node.children, key=uct_value)

    def _compute_volume_score(self, state: ModelState) -> float:
        """GoT: Compute score based on thought volume"""
        try:
            if not self.thought_graph.has_node(state.id):
                return 0.0
            volume = len(nx.ancestors(self.thought_graph, state.id))
            return 0.1 * volume  # Scale factor can be adjusted
        except Exception as e:
            logger.error(f"Error computing volume score: {e}")
            return 0.0

    def _compute_latency_penalty(self, state: ModelState) -> float:
        """GoT: Compute penalty based on thought latency"""
        try:
            if not self.thought_graph.has_node(state.id):
                return 0.0
            
            paths = []
            root_nodes = [n for n in self.thought_graph.nodes() 
                         if self.thought_graph.in_degree(n) == 0]
                         
            for root in root_nodes:
                if nx.has_path(self.thought_graph, root, state.id):
                    paths.extend(nx.all_simple_paths(self.thought_graph, root, state.id))
            
            if not paths:
                return 0.0
                
            max_path_length = max(len(path) for path in paths)
            return 0.05 * max_path_length  # Scale factor can be adjusted
            
        except Exception as e:
            logger.error(f"Error computing latency penalty: {e}")
            return 0.0

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

    def aggregate_thoughts(self, nodes: List[MCTSNode]) -> Optional[MCTSNode]:
        """GoT: Aggregate multiple thoughts into a new one"""
        if not nodes:
            return None
            
        try:
            # Combine equations using the most complex one as base
            base_node = max(nodes, key=lambda n: len(n.state.equations[0].split()))
            new_state = base_node.state.copy()
            
            # Track parentage
            new_node = MCTSNode(new_state, parent=None)
            new_node.thought_parents.extend(nodes)
            
            # Add to thought graph
            self._add_to_thought_graph(new_node)
            
            return new_node
            
        except Exception as e:
            logger.error(f"Error aggregating thoughts: {e}")
            return None

    def refine_thought(self, node: MCTSNode) -> Optional[MCTSNode]:
        """GoT: Refine a thought through iteration"""
        try:
            new_state = self._apply_refinement(node.state)
            new_node = MCTSNode(new_state, parent=node)
            new_node.thought_parents = [node]
            new_node.refinement_count = node.refinement_count + 1
            
            # Add refinement to thought graph
            self._add_to_thought_graph(new_node)
            # Add self-loop for refinement
            self.thought_graph.add_edge(new_node.state.id, new_node.state.id)
            
            return new_node
            
        except Exception as e:
            logger.error(f"Error refining thought: {e}")
            return None

    def _apply_refinement(self, state: ModelState) -> ModelState:
        """Apply refinement transformation to a state"""
        # Simple refinement: add a small random perturbation to parameters
        new_state = state.copy()
        for param in new_state.parameters:
            current_value = new_state.parameters[param]
            # Add small random adjustment (±10%)
            adjustment = current_value * (1 + np.random.uniform(-0.1, 0.1))
            new_state.parameters[param] = adjustment
        return new_state

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

    def get_thought_metrics(self, state: ModelState) -> Dict:
        """Get GoT metrics for a thought"""
        return {
            'volume': len(nx.ancestors(self.thought_graph, state.id)) if self.thought_graph.has_node(state.id) else 0,
            'latency': self._compute_latency_penalty(state)
        }