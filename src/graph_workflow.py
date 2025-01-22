# src/graph_workflow.py

# - Manages the step-by-step process of discovering models
# - Controls when to generate, evaluate, and modify models
# - Includes steps for combining promising models
# - Tracks how models relate to each other

from typing import Dict, List, TypedDict, Annotated, Optional
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
import operator
from enum import Enum
import logging
from dataclasses import dataclass
import networkx as nx
from .core import ModelState
from .knowledge_graph import CognitiveKnowledgeGraph
from .transformations import ThoughtTransformations

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State maintained in the graph"""
    current_model: ModelState
    knowledge: Dict
    messages: List[BaseMessage]
    next_step: str
    metrics: Dict

class ModelDiscoveryGraph:
    def __init__(self, knowledge_graph: CognitiveKnowledgeGraph):
        self.kg = knowledge_graph
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7,
        )
        # Graph structure for tracking thoughts
        self.thought_graph = nx.DiGraph()
        self.transformations = ThoughtTransformations(self.llm)
        
        # Init metrics structure
        self.default_metrics = {
            'scores': [],
            'model_complexity': [],
            'iterations': [],
            'exploration_paths': [],
            'thought_volumes': [],  # GoT metric
            'thought_latencies': [] # GoT metric
        }

    async def run_workflow(self, state: AgentState) -> AgentState:
        """Run workflow steps manually to avoid recursion"""
        try:
            # Execute workflow steps sequentially
            state = await self.query_knowledge_node(state)
            state = await self.generate_hypothesis_node(state)
            
            # GoT operations, update!
            state = await self.aggregate_thoughts_node(state)
            state = await self.refine_thought_node(state)
            
            state = await self.evaluate_model_node(state)
            state = await self.update_knowledge_node(state)
            state = await self.check_convergence_node(state)
            
            # Compute GoT metrics
            metrics = self.compute_thought_metrics(state["current_model"])
            state["metrics"]["thought_volumes"].append(metrics["volume"])
            state["metrics"]["thought_latencies"].append(metrics["latency"])
            
            # Check if we should continue or end :D
            next_step = self.decide_next_step(state)
            if next_step == "complete":
                state = await self.end_workflow_node(state)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in workflow execution: {e}")
            return state

    async def aggregate_thoughts_node(self, state: AgentState) -> AgentState:
        """GoT: Node for aggregating multiple thoughts"""
        try:
            # Get recent models from graph
            recent_models = list(self.thought_graph.nodes())[-3:]  # Last 3 thoughts
            if len(recent_models) > 1:
                aggregated_model = await self.transformations.aggregate_thoughts(recent_models)
                if aggregated_model:
                    # Add to graph with edges from parents
                    self.thought_graph.add_node(aggregated_model)
                    for model in recent_models:
                        self.thought_graph.add_edge(model, aggregated_model)
                    state["current_model"] = aggregated_model
        except Exception as e:
            logger.error(f"Error in aggregate_thoughts_node: {e}")
        return state

    async def refine_thought_node(self, state: AgentState) -> AgentState:
        """GoT: Node for refining a thought"""
        try:
            refined_model = await self.transformations.refine_thought(state["current_model"])
            if refined_model:
                # Add refinement to graph with self-loop
                self.thought_graph.add_node(refined_model)
                self.thought_graph.add_edge(state["current_model"], refined_model)
                self.thought_graph.add_edge(refined_model, refined_model)  # Self-loop
                state["current_model"] = refined_model
        except Exception as e:
            logger.error(f"Error in refine_thought_node: {e}")
        return state

    def compute_thought_metrics(self, thought: ModelState) -> Dict:
        """GoT: Compute volume and latency metrics for a thought"""
        try:
            if not self.thought_graph.has_node(thought):
                return {"volume": 0, "latency": 0}
            
            # Volume: number of predecessor thoughts
            volume = len(nx.ancestors(self.thought_graph, thought))
            
            # Latency: longest path to this thought
            root_nodes = [n for n in self.thought_graph.nodes() 
                         if self.thought_graph.in_degree(n) == 0]
            
            if not root_nodes:
                return {"volume": volume, "latency": 0}
                
            paths = []
            for root in root_nodes:
                if nx.has_path(self.thought_graph, root, thought):
                    paths.extend(nx.all_simple_paths(self.thought_graph, root, thought))
            
            latency = max(len(path) for path in paths) if paths else 0
            
            return {"volume": volume, "latency": latency}
            
        except Exception as e:
            logger.error(f"Error computing thought metrics: {e}")
            return {"volume": 0, "latency": 0}

    async def query_knowledge_node(self, state: AgentState) -> AgentState:
        """Node for querying knowledge graph"""
        try:
            current_mechanism = self._extract_mechanism(state["current_model"])
            knowledge = self.kg.query_mechanism(current_mechanism)
            state["knowledge"] = knowledge
            
            # Add initial thought to graph if not present
            if not self.thought_graph.has_node(state["current_model"]):
                self.thought_graph.add_node(state["current_model"])
        except Exception as e:
            logger.error(f"Error in query_knowledge_node: {e}")
        return state
    
    async def generate_hypothesis_node(self, state: AgentState) -> AgentState:
        """Node for generating new model hypotheses"""
        try:
            system_message = "You are an expert in cognitive modeling. Generate new hypotheses based on known mechanisms."
            prompt = self._create_hypothesis_prompt(state)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            response = await self.llm.agenerate([messages])
            new_model = self._parse_llm_response(response.generations[0][0].text)
            if new_model:
                # Add to thought graph with edge from current model
                self.thought_graph.add_node(new_model)
                self.thought_graph.add_edge(state["current_model"], new_model)
                state["current_model"] = new_model
        except Exception as e:
            logger.error(f"Error in generate_hypothesis_node: {e}")
        return state

   
    async def evaluate_model_node(self, state: AgentState) -> AgentState:
        """Node for evaluating models"""
        try:
            score = self._compute_model_score(state["current_model"], state["knowledge"])
            state["current_model"].score = score
        except Exception as e:
            logger.error(f"Error in evaluate_model_node: {e}")
        return state
    
    async def update_knowledge_node(self, state: AgentState) -> AgentState:
        """Node for updating knowledge graph"""
        try:
            if state["current_model"].score:
                self.kg.add_model_knowledge(
                    self._extract_mechanism(state["current_model"]),
                    state["current_model"]
                )
        except Exception as e:
            logger.error(f"Error in update_knowledge_node: {e}")
        return state
    
    async def check_convergence_node(self, state: AgentState) -> AgentState:
        """Node for checking convergence"""
        try:
            if "metrics" not in state:
                state["metrics"] = self.default_metrics.copy()
            
            state["metrics"]["scores"].append(state["current_model"].score)
            state["metrics"]["model_complexity"].append(len(state["current_model"].equations[0].split()))
            state["metrics"]["iterations"].append(len(state["metrics"]["scores"]))
            if state["current_model"].equations:
                state["metrics"]["exploration_paths"].append(state["current_model"].equations[0])
        except Exception as e:
            logger.error(f"Error in check_convergence_node: {e}")
        return state
    
    async def end_workflow_node(self, state: AgentState) -> AgentState:
        """Final node in the workflow"""
        state["next_step"] = "complete"
        return state

    def decide_next_step(self, state: AgentState) -> str:
        """Decide next step based on current state"""
        try:
            if len(state["metrics"]["scores"]) >= 50:
                return "complete"
            if state["current_model"].score > 0.9:
                return "complete"
            if len(state["metrics"]["scores"]) > 5:
                recent_scores = state["metrics"]["scores"][-5:]
                if max(recent_scores) - min(recent_scores) < 0.01:
                    return "refine"
            return "continue"
        except Exception as e:
            logger.error(f"Error in decide_next_step: {e}")
            return "continue"

    def _create_hypothesis_prompt(self, state: AgentState) -> str:
        """Create prompt for hypothesis generation"""
        return f"""
        Current model:
        {state["current_model"].equations[0]}
        
        Known mechanisms:
        {state["knowledge"]}
        
        Generate a new cognitive model that:
        1. Builds on successful aspects of previous models
        2. Incorporates relevant mechanisms
        3. Is mathematically precise
        4. Is theoretically sound
        
        RESPONSE FORMAT:
        EQUATION: [equation]
        PARAMETERS: [param1: value1, param2: value2, ...]
        THEORETICAL_BASIS: [brief explanation]
        """

    def _parse_llm_response(self, text: str) -> Optional[ModelState]:
        """Parse LLM response into a ModelState"""
        try:
            lines = text.strip().split('\n')
            equation = None
            parameters = {}
            
            for line in lines:
                if line.startswith('EQUATION:'):
                    equation = line.replace('EQUATION:', '').strip()
                elif line.startswith('PARAMETERS:'):
                    params_str = line.replace('PARAMETERS:', '').strip()
                    for pair in params_str.split(','):
                        if ':' in pair:
                            key, value_str = pair.split(':')
                            key = key.strip()
                            try:
                                value = float(value_str.strip())
                                parameters[key] = value
                            except ValueError:
                                continue
            
            if equation and parameters:
                return ModelState(equations=[equation], parameters=parameters)
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            
        return None

    def _compute_model_score(self, model: ModelState, knowledge: Dict) -> float:
        """Compute model score"""
        try:
            # Simple scoring based on equation complexity and parameter count
            complexity_penalty = len(model.equations[0].split()) * 0.01
            param_penalty = len(model.parameters) * 0.05
            base_score = 0.5  # Base score
            
            return base_score - complexity_penalty - param_penalty
            
        except Exception as e:
            logger.error(f"Error computing model score: {e}")
            return 0.0

    def _extract_mechanism(self, model: ModelState) -> str:
        """Extract mechanism type from model"""
        if "Q(t)" in model.equations[0]:
            return "reinforcement_learning"
        if "WM(t)" in model.equations[0]:
            return "working_memory"
        return "unknown_mechanism"