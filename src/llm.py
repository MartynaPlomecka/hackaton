# src/llm.py:

# - Handles communication with the language model
# - Formats prompts and parses responses

import os
from typing import Dict, List, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import numpy as np
from .core import ModelState
from .knowledge_graph import CognitiveKnowledgeGraph

load_dotenv()

class EnhancedLLMInterface:
    """Enhanced LLM interface with knowledge graph integration"""
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.kg = CognitiveKnowledgeGraph()
        if not use_mock:
            self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.chat_model = ChatOpenAI(model="gpt-4-turbo-preview")
    
    async def generate(self, state: ModelState, context: Optional[Dict] = None) -> ModelState:
        """Generate new model variations using LLM and knowledge graph"""
        if self.use_mock:
            return self._mock_generate(state)
        else:
            return await self._llm_generate(state, context)

    async def _llm_generate(self, state: ModelState, context: Optional[Dict]) -> ModelState:
        """Use OpenAI API with knowledge graph context"""
        # Get relevant knowledge from graph
        mechanism_info = self.kg.query_mechanism("reinforcement_learning")
        related_mechanisms = self.kg.get_related_mechanisms("reinforcement_learning")
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert in cognitive modeling and 
            reinforcement learning. Generate variations of cognitive models that are 
            theoretically sound and mathematically valid."""),
            HumanMessage(content=self._create_detailed_prompt(state, mechanism_info, related_mechanisms))
        ])
        
        try:
            response = await self.chat_model.ainvoke(prompt)
            return self._parse_llm_response(response.content, state)
        except Exception as e:
            print(f"LLM generation error: {e}")
            return self._mock_generate(state)

    def _create_detailed_prompt(self, state: ModelState, mechanism_info: Dict, related_mechanisms: List) -> str:
        """Create detailed prompt using knowledge graph information"""
        return f"""
        Current cognitive model equation(s):
        {state.equations[0]}
        
        Current parameters:
        {state.parameters}
        
        Relevant mechanism information:
        {mechanism_info}
        
        Related mechanisms to consider:
        {related_mechanisms}
        
        Generate a variation of this model that:
        1. Incorporates known cognitive mechanisms
        2. Is mathematically precise
        3. Includes appropriate parameters
        4. Could explain human learning in a two-armed bandit task
        
        YOU MUST FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:
        EQUATION: [your equation]
        PARAMETERS: [parameter1: value1, parameter2: value2, ...]
        THEORETICAL_BASIS: [brief explanation of theoretical justification]
        """

    def _parse_llm_response(self, response_content: str, original_state: ModelState) -> ModelState:
        """Parse LLM response with enhanced validation"""
        try:
            lines = [line.strip() for line in response_content.split('\n') if line.strip()]
            equation = None
            parameters = {}
            theoretical_basis = None
            
            for line in lines:
                if line.startswith('EQUATION:'):
                    equation = line.replace('EQUATION:', '').strip()
                elif line.startswith('PARAMETERS:'):
                    params_str = line.replace('PARAMETERS:', '').strip()
                    param_pairs = params_str.split(',')
                    for pair in param_pairs:
                        if ':' in pair:
                            key, value = pair.split(':')
                            try:
                                parameters[key.strip()] = float(value.strip())
                            except ValueError:
                                print(f"Error parsing parameter value: {pair}")
                elif line.startswith('THEORETICAL_BASIS:'):
                    theoretical_basis = line.replace('THEORETICAL_BASIS:', '').strip()
            
            if equation and parameters:
                new_state = ModelState(
                    equations=[equation],
                    parameters=parameters
                )
                if self.validate_equation(new_state):
                    return new_state
                    
            print("Failed to generate valid model")
            return original_state.copy()
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return original_state.copy()

    def validate_equation(self, state: ModelState) -> bool:
        """Enhanced validation of generated equations"""
        equation = state.equations[0]
        try:
            if equation.count('(') != equation.count(')'):
                return False
                
            # Check for required components
            required_components = ['Q(t', 'R(t)']
            if not all(comp in equation for comp in required_components):
                return False
                
            # Validate parameters
            equation_params = set()
            for param in ['alpha', 'beta', 'gamma', 'temp']:
                if param in equation:
                    equation_params.add(param)
                    
            if not equation_params.issubset(set(state.parameters.keys())):
                return False
                
            # Check parameter ranges
            for param, value in state.parameters.items():
                if param in ['alpha', 'gamma'] and (value < 0 or value > 1):
                    return False
                if param == 'temp' and value <= 0:
                    return False
                    
            return True
            
        except Exception:
            return False

    def _mock_generate(self, state: ModelState) -> ModelState:
        """mock generation with theoretical components, TO BE UPD"""
        new_state = state.copy()
        
        modifications = [
            (" + alpha * PE", {"alpha": 0.1}, "Standard RL update"),
            (" * (1 - gamma)", {"gamma": 0.1}, "Memory decay"),
            (" / temp", {"temp": 1.0}, "Action selection"),
            (" + beta * RPE", {"beta": 0.2}, "Surprise modulation"),
        ]
        
        if len(new_state.equations[0].split('+')) < 3:
            mod, params, _ = np.random.choice(modifications)
            new_state.equations[0] += mod
            new_state.parameters.update(params)
            
        return new_state