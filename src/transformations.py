#  src/transformations.py:

# - Handles combining multiple models into better ones
# - Uses LLM to refine models
# - Creates prompts for model improvement
# - Fallback strategies when LLM suggestions fail


import numpy as np
from typing import List
from .core import ModelState

class ThoughtTransformations:
    def __init__(self, llm):
        self.llm = llm

    async def aggregate_thoughts(self, thoughts: List[ModelState]) -> ModelState:
        """Aggregate multiple thoughts into a new enhanced thought"""
        try:
            # Create aggregation prompt
            prompt = self._create_aggregation_prompt([t.equations[0] for t in thoughts])
            response = await self.llm.agenerate([[{"role": "user", "content": prompt}]])
            new_equation = self._parse_equation(response.generations[0][0].text)
            
            # Combine parameters
            combined_params = {}
            for thought in thoughts:
                combined_params.update(thought.parameters)
            
            return ModelState(
                equations=[new_equation] if new_equation else thoughts[0].equations,
                parameters=combined_params
            )
            
        except Exception as e:
            print(f"Error in aggregate_thoughts: {e}")
            # Fallback to most complex thought if aggregation fails
            return max(thoughts, key=lambda t: len(t.equations[0].split())).copy()

    async def refine_thought(self, thought: ModelState) -> ModelState:
        """Refine a thought through iteration"""
        try:
            prompt = self._create_refinement_prompt(thought.equations[0])
            response = await self.llm.agenerate([[{"role": "user", "content": prompt}]])
            refined_equation = self._parse_equation(response.generations[0][0].text)
            
            # If LLM refinement fails, fall back to parameter adjustment
            if not refined_equation:
                return self._apply_parameter_refinement(thought)
            
            return ModelState(
                equations=[refined_equation],
                parameters=thought.parameters.copy()
            )
            
        except Exception as e:
            print(f"Error in refine_thought: {e}")
            return self._apply_parameter_refinement(thought)

    def _apply_parameter_refinement(self, thought: ModelState) -> ModelState:
        """Apply small random adjustments to parameters"""
        new_state = thought.copy()
        for param in new_state.parameters:
            current_value = new_state.parameters[param]
            # Add small random adjustment (±10%)
            adjustment = current_value * (1 + np.random.uniform(-0.1, 0.1))
            new_state.parameters[param] = adjustment
        return new_state

    def _create_aggregation_prompt(self, equations: List[str]) -> str:
        """Create prompt for LLM to combine equations"""
        return f"""Given these cognitive model equations:
        {', '.join(equations)}
        
        Create a new single equation that combines their strengths while addressing their weaknesses.
        The new equation should be mathematically valid and theoretically sound.
        Maintain the Q(t) and R(t) notation style.
        
        RESPONSE FORMAT:
        EQUATION: [your combined equation]
        """

    def _create_refinement_prompt(self, equation: str) -> str:
        """Create prompt for LLM to refine equation"""
        return f"""Consider this cognitive model equation:
        {equation}
        
        Improve this equation to make it more elegant and theoretically sound.
        Maintain mathematical validity and Q(t), R(t) notation style.
        
        RESPONSE FORMAT:
        EQUATION: [your refined equation]
        """

    def _parse_equation(self, text: str) -> str:
        """Parse equation from LLM response"""
        try:
            for line in text.split('\n'):
                if line.strip().startswith('EQUATION:'):
                    equation = line.replace('EQUATION:', '').strip()
                    # Basic validation that it contains Q(t)
                    if 'Q(t)' in equation:
                        return equation
            return ""
        except Exception:
            return ""