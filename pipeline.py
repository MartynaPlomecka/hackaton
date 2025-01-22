# `pipeline.py`:

# - coordinates everything
# - Balances exploration vs refinement
# - Keeps track of the best models found
# - Manages a pool of promising models


import asyncio
from src.core import ModelState, generate_test_data
from src.mcts_kg import EnhancedMCTS, MCTSNode
from src.graph_workflow import ModelDiscoveryGraph, AgentState
from src.evaluation import SimpleEvaluator
from src.llm import EnhancedLLMInterface
from src.knowledge_graph import CognitiveKnowledgeGraph
from src.transformations import ThoughtTransformations
import numpy as np
from typing import Optional, Tuple, List
import json
from datetime import datetime
import logging
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('discovery.log')
    ]
)
logger = logging.getLogger(__name__)

class EnhancedModelDiscoveryPipeline:
    def __init__(self, 
                 use_mock_llm: bool = True,
                 n_iterations: int = 50,
                 exploration_constant: float = 1.414):
        """Initialize the enhanced pipeline with both MCTS and workflow components"""
        logger.info("Initializing Enhanced Model Discovery Pipeline...")
        
        self.test_data = generate_test_data(n_trials=100)
        self.kg = CognitiveKnowledgeGraph()
        self.mcts = EnhancedMCTS(self.kg)
        self.evaluator = SimpleEvaluator(self.test_data)
        self.llm = EnhancedLLMInterface(use_mock=use_mock_llm)
        self.graph_workflow = ModelDiscoveryGraph(self.kg)
        self.n_iterations = n_iterations
        
        # Thought pool for GoT
        self.thought_pool: List[ModelState] = []
        self.max_thought_pool_size = 10
        
        # Tracking best models
        self.best_score = float('-inf')
        self.best_model: Optional[ModelState] = None
        
        # metrics tracking with GoT metrics
        self.metrics = {
            'scores': [],
            'model_complexity': [],
            'iterations': [],
            'exploration_paths': [],
            'thought_volumes': [],  # GoT metric
            'thought_latencies': [], # GoT metric
            'aggregation_counts': [], # Track number of thought aggregations
            'refinement_counts': []  # Track number of thought refinements
        }
        
        # State tracking
        self.is_running = True
        
    def setup_signal_handlers(self):
        """handlers for shutdown"""
        def handle_interrupt(signum, frame):
            logger.info("\nReceived interrupt signal. shutdown...")
            self.is_running = False
            self.save_results()
            
        signal.signal(signal.SIGINT, handle_interrupt)
        signal.signal(signal.SIGTERM, handle_interrupt)

    def _manage_thought_pool(self, new_thought: ModelState):
        """Manage the thought pool, keeping only the best thoughts"""
        self.thought_pool.append(new_thought)
        if len(self.thought_pool) > self.max_thought_pool_size:
            # Keep only the best thoughts based on scores
            self.thought_pool.sort(key=lambda x: x.score if x.score is not None else float('-inf'), 
                                 reverse=True)
            self.thought_pool = self.thought_pool[:self.max_thought_pool_size]
        
    async def run(self):
        """Run the discovery pipeline with GoT enhancements"""
        try:
            # Initial model state
            initial_state = ModelState(
                equations=["Q(t+1) = Q(t) + (R(t) - Q(t))"],
                parameters={}
            )
            
            # Create root node for MCTS
            root = MCTSNode(initial_state)
            self._manage_thought_pool(initial_state)
            
            # Init workflow state
            workflow_state = AgentState(
                current_model=initial_state,
                knowledge={},
                messages=[],
                next_step="start",
                metrics=self.metrics.copy()
            )
            
            logger.info("Starting model discovery process...")
            
            for i in range(self.n_iterations):
                if not self.is_running:
                    logger.info("Stopping discovery process due to interrupt...")
                    break
                    
                # MCTS Selection with knowledge guidance
                node = self.mcts.select_node(root)
                
                # MCTS Expansion
                if node.untried_actions:
                    child = self.mcts.expand(node)
                    if child:
                        node = child
                        self.metrics['exploration_paths'].append(child.state.equations[0])
                
                # Update workflow state with current MCTS node
                workflow_state["current_model"] = node.state
                
                try:
                    # Run workflow using manual execution
                    workflow_state = await self.graph_workflow.run_workflow(workflow_state)
                    
                    simulation_state = workflow_state["current_model"]
                    self._manage_thought_pool(simulation_state)
                    
                    # Eval
                    score = self.evaluator.evaluate_model(simulation_state)
                    
                    # Update metrics consistently
                    self._update_metrics(simulation_state, score, i)
                    
                    # Update best model if needed
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = simulation_state.copy()
                        logger.info(f"\nNew best model found (iteration {i+1}):")
                        self._print_model_details(simulation_state, score)
                    
                    # Get GoT metrics
                    got_metrics = self.graph_workflow.compute_thought_metrics(simulation_state)
                    self.metrics['thought_volumes'].append(got_metrics['volume'])
                    self.metrics['thought_latencies'].append(got_metrics['latency'])
                    
                    # Backprop for MCTS with enhanced scoring
                    enhanced_score = self._compute_enhanced_score(score, got_metrics)
                    self._backpropagate(node, enhanced_score)
                    
                    # Checking for workflow completion
                    if workflow_state["next_step"] == "complete":
                        logger.info("Workflow signaled completion")
                        break
                    
                except Exception as e:
                    logger.error(f"Error in iteration {i}: {str(e)}")
                    continue
                
                # Progress update
                if (i + 1) % 10 == 0:
                    self._print_progress_update(i + 1)
                    
        except Exception as e:
            logger.error(f"Critical error in pipeline: {str(e)}")
            raise
        
        finally:
            # Saving final results
            logger.info("\nDiscovery process complete!")
            if self.best_model:
                logger.info("\nBest model found:")
                self._print_model_details(self.best_model, self.best_score)
                self.save_results()
            self._save_metrics()

    def _compute_enhanced_score(self, base_score: float, got_metrics: dict) -> float:
        """Compute enhanced score incorporating GoT metrics"""
        volume_bonus = 0.1 * got_metrics['volume']  # Reward thoughts with high volume
        latency_penalty = 0.05 * got_metrics['latency']  # Penalize high latency
        return base_score + volume_bonus - latency_penalty

    def _update_metrics(self, state: ModelState, score: float, iteration: int):
        """Update all metrics including GoT-specific ones"""
        self.metrics['scores'].append(score)
        self.metrics['model_complexity'].append(len(state.equations[0].split()))
        self.metrics['iterations'].append(iteration)

    def _backpropagate(self, node: MCTSNode, score: float):
        """Backpropagate results through the tree"""
        while node:
            node.visits += 1
            node.value += score
            node = node.parent

    def _print_model_details(self, state: ModelState, score: float):
        """Print detailed model information including GoT metrics"""
        print("\nEquation:")
        print(f"  {state.equations[0]}")
        print("\nParameters:")
        for param, value in state.parameters.items():
            print(f"  {param}: {value:.3f}")
        print(f"\nScore: {score:.3f}")
        
        # Add GoT metrics
        got_metrics = self.graph_workflow.compute_thought_metrics(state)
        print(f"Thought Volume: {got_metrics['volume']}")
        print(f"Thought Latency: {got_metrics['latency']}")
        print(f"Model Complexity: {len(state.equations[0].split())}")

    def _print_progress_update(self, iteration: int):
        """Print detailed progress update with GoT metrics"""
        print(f"\nCompleted iteration {iteration}/{self.n_iterations}")
        if self.metrics['scores']:
            avg_score = np.mean(self.metrics['scores'][-10:])
            print(f"Average score (last 10): {avg_score:.3f}")
            print(f"Best score so far: {self.best_score:.3f}")
            print(f"Average thought volume: {np.mean(self.metrics['thought_volumes'][-10:]):.2f}")
            print(f"Average thought latency: {np.mean(self.metrics['thought_latencies'][-10:]):.2f}")

    def save_results(self):
        """Save comprehensive results to a JSON file"""
        if not self.best_model:
            return
            
        got_metrics = self.graph_workflow.compute_thought_metrics(self.best_model)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'best_model': {
                'equations': self.best_model.equations,
                'parameters': {k: float(v) for k, v in self.best_model.parameters.items()},
                'score': float(self.best_score),
                'thought_volume': got_metrics['volume'],
                'thought_latency': got_metrics['latency']
            },
            'knowledge_graph_stats': self.kg.get_graph_statistics(),
            'settings': {
                'n_iterations': self.n_iterations,
                'exploration_constant': self.mcts.exploration_constant,
                'use_mock_llm': self.llm.use_mock
            }
        }
        
        filename = f"model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {filename}")

    def _save_metrics(self):
        """Save detailed metrics including GoT metrics to a separate file"""
        metrics_file = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'scores': [float(s) for s in self.metrics['scores']],
                'model_complexity': self.metrics['model_complexity'],
                'iterations': self.metrics['iterations'],
                'exploration_paths': self.metrics['exploration_paths'][-10:],
                'thought_volumes': self.metrics['thought_volumes'],
                'thought_latencies': self.metrics['thought_latencies'],
                'aggregation_counts': self.metrics['aggregation_counts'],
                'refinement_counts': self.metrics['refinement_counts']
            }, f, indent=2)
        logger.info(f"Detailed metrics saved to {metrics_file}")

async def main():
    """Main entry point"""
    try:
        pipeline = EnhancedModelDiscoveryPipeline(
            use_mock_llm=False,  
            n_iterations=50,
            exploration_constant=1.414
        )
        
        pipeline.setup_signal_handlers()
        
        await pipeline.run()
        
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        if hasattr(pipeline, 'save_results'):
            pipeline.save_results()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())