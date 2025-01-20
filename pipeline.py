import asyncio
from src.core import ModelState, generate_test_data
from src.mcts_kg import EnhancedMCTS, MCTSNode
from src.graph_workflow import ModelDiscoveryGraph, AgentState
from src.evaluation import SimpleEvaluator
from src.llm import EnhancedLLMInterface
from src.knowledge_graph import CognitiveKnowledgeGraph
import numpy as np
from typing import Optional, Tuple
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
        
        # Tracking best models
        self.best_score = float('-inf')
        self.best_model: Optional[ModelState] = None
        
        # Metrics tracking 
        self.metrics = {
            'scores': [],
            'model_complexity': [],
            'iterations': [],
            'exploration_paths': []
        }
        
        # State tracking
        self.is_running = True
        
    def setup_signal_handlers(self):
        """handlers forshutdown"""
        def handle_interrupt(signum, frame):
            logger.info("\nReceived interrupt signal. shutdown...")
            self.is_running = False
            self.save_results()
            
        signal.signal(signal.SIGINT, handle_interrupt)
        signal.signal(signal.SIGTERM, handle_interrupt)
        
    async def run(self):
        """Run the discovery pipeline"""
        try:
            # Initial model state
            initial_state = ModelState(
                equations=["Q(t+1) = Q(t) + (R(t) - Q(t))"],
                parameters={}
            )
            
            # Create root node for MCTS
            root = MCTSNode(initial_state)
            
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
                    
                    # Eval
                    score = self.evaluator.evaluate_model(simulation_state)
                    
                    # Update metrics consistently
                    self.metrics['scores'].append(score)
                    self.metrics['model_complexity'].append(len(simulation_state.equations[0].split()))
                    self.metrics['iterations'].append(i)
                    
                    # Update best model if needed
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = simulation_state.copy()
                        logger.info(f"\nNew best model found (iteration {i+1}):")
                        self._print_model_details(simulation_state, score)
                    
                    # Backprop for MCTS
                    self._backpropagate(node, score)
                    
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
            # Savinh final results
            logger.info("\nDiscovery process complete!")
            if self.best_model:
                logger.info("\nBest model found:")
                self._print_model_details(self.best_model, self.best_score)
                self.save_results()
            self._save_metrics()

    def _backpropagate(self, node: MCTSNode, score: float):
        """Backpropagate results through the tree"""
        while node:
            node.visits += 1
            node.value += score
            node = node.parent

    def _print_model_details(self, state: ModelState, score: float):
        """Print detailed model information"""
        print("\nEquation:")
        print(f"  {state.equations[0]}")
        print("\nParameters:")
        for param, value in state.parameters.items():
            print(f"  {param}: {value:.3f}")
        print(f"\nScore: {score:.3f}")
        print(f"Model Complexity: {len(state.equations[0].split())}")

    def _print_progress_update(self, iteration: int):
        """Print detailed progress update"""
        print(f"\nCompleted iteration {iteration}/{self.n_iterations}")
        if self.metrics['scores']:
            avg_score = np.mean(self.metrics['scores'][-10:])
            print(f"Average score (last 10): {avg_score:.3f}")
            print(f"Best score so far: {self.best_score:.3f}")

    def save_results(self):
        """Save comprehensive results to a JSON file"""
        if not self.best_model:
            return
            
        results = {
            'timestamp': datetime.now().isoformat(),
            'best_model': {
                'equations': self.best_model.equations,
                'parameters': {k: float(v) for k, v in self.best_model.parameters.items()},
                'score': float(self.best_score)
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
        """Save detailed metrics to a separate file"""
        metrics_file = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'scores': [float(s) for s in self.metrics['scores']],
                'model_complexity': self.metrics['model_complexity'],
                'iterations': self.metrics['iterations'],
                'exploration_paths': self.metrics['exploration_paths'][-10:]
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