import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import yaml
from stable_baselines3 import PPO

class VectorDatabase:
    def __init__(self):
        self.embeddings = []
        self.metadata = []
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def add_entry(self, text: str, metadata: Dict[str, Any]):
        embedding = self.encoder.encode(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata)
        
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.encoder.encode(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.metadata[i] for i in top_indices]

class SmartCoach:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vector_db = VectorDatabase()
        self._load_knowledge_base()
        
        self.llm_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(config['coach']['llm_model'])
        self.llm = AutoModel.from_pretrained(config['coach']['llm_model']).to(self.llm_device)
        
        self.performance_history = {
            'episode_rewards': [],
            'actions_taken': [],
            'adjustments_made': []
        }
    
    def _load_knowledge_base(self):
        """Load RAG knowledge base from files"""
        kb_paths = [
            "knowledge/rl_trading_kb.json",
            os.path.join(os.path.dirname(__file__), "../knowledge/rl_trading_kb.json")
        ]
        
        for path in kb_paths:
            if os.path.exists(path):
                with open(path) as f:
                    kb_data = json.load(f)
                    for item in kb_data:
                        self.vector_db.add_entry(
                            text=item['description'],
                            metadata=item
                        )
                break
    
    def analyze_performance(self, episode_rewards: List[float], 
                          current_config: Dict[str, Any], 
                          model: PPO) -> Optional[Dict[str, Any]]:
        """Analyze performance and suggest adjustments"""
        self.performance_history['episode_rewards'] = episode_rewards
        
        recent_rewards = episode_rewards[-20:] if len(episode_rewards) > 20 else episode_rewards
        avg_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        
       
        query = f"""
        RL trading agent performance:
        - Average reward: {avg_reward:.2f}
        - Reward std: {std_reward:.2f}
        - Recent trend: {'increasing' if avg_reward > np.mean(episode_rewards[:-20]) else 'decreasing'}
        """
        
        similar_cases = self.vector_db.search(query)
        
        prompt = self._build_llm_prompt(
            performance_summary={
                'avg_reward': avg_reward,
                'std_reward': std_reward,
                'recent_trend': 'increasing' if avg_reward > np.mean(episode_rewards[:-20]) else 'decreasing'
            },
            current_config=current_config,
            similar_cases=similar_cases
        )
        
        recommendation = self._get_llm_recommendation(prompt)
        
        if self._validate_recommendation(recommendation):
            self._apply_adjustments(recommendation, model, current_config)
            self.performance_history['adjustments_made'].append(recommendation)
            return recommendation
        
        return None
    
    def _build_llm_prompt(self, performance_summary: Dict[str, Any],
                         current_config: Dict[str, Any],
                         similar_cases: List[Dict[str, Any]]) -> str:
        """Construct detailed prompt for LLM"""
        similar_cases_str = "\n".join(
            f"Case {i+1}:\n- Description: {case['description']}\n- Solution: {case['solution']}"
            for i, case in enumerate(similar_cases))
        
        return f"""
        You are an expert RL trading coach. Analyze the current performance and suggest parameter adjustments.
        
        Current Performance:
        - Average Reward: {performance_summary['avg_reward']:.2f}
        - Reward Std: {performance_summary['std_reward']:.2f}
        - Recent Trend: {performance_summary['recent_trend']}
        
        Current Configuration:
        {json.dumps(current_config['ppo'], indent=2)}
        
        Similar Historical Cases:
        {similar_cases_str}
        
        Guidelines:
        1. Only suggest small, incremental changes
        2. Prioritize adjustments to learning rate, entropy coefficient, and clip range first
        3. Explain your reasoning
        4. Output in JSON format with: adjustment_type, parameter, new_value, reasoning
        
        Recommendation:
        """
    
    def _get_llm_recommendation(self, prompt: str) -> Dict[str, Any]:
        """Get structured recommendation from LLM"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_device)
        outputs = self.llm.generate(**inputs, max_new_tokens=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            json_str = response.split('{', 1)[1].rsplit('}', 1)[0]
            json_str = '{' + json_str + '}'
            return json.loads(json_str)
        except:
            print(f"Failed to parse LLM response: {response}")
            return None
    
    def _validate_recommendation(self, recommendation: Dict[str, Any]) -> bool:
        """Validate the LLM recommendation"""
        if not recommendation:
            return False
            
        valid_params = [
            'learning_rate', 'ent_coef', 'clip_range', 
            'gamma', 'gae_lambda', 'n_steps'
        ]
        
        if recommendation.get('parameter') not in valid_params:
            return False
            
        param = recommendation['parameter']
        new_value = recommendation['new_value']
        
        ranges = {
            'learning_rate': (1e-6, 1e-2),
            'ent_coef': (0.0, 0.2),
            'clip_range': (0.1, 0.3),
            'gamma': (0.9, 0.999),
            'gae_lambda': (0.9, 1.0),
            'n_steps': (256, 2048)
        }
        
        return ranges[param][0] <= new_value <= ranges[param][1]
    
    def _apply_adjustments(self, recommendation: Dict[str, Any],
                         model: PPO, config: Dict[str, Any]):
        """Apply the validated adjustments"""
        param = recommendation['parameter']
        new_value = recommendation['new_value']
        
        if param == 'learning_rate':
            model.learning_rate = new_value
        elif param == 'ent_coef':
            model.ent_coef = new_value
        elif param == 'clip_range':
            model.clip_range = new_value
        elif param == 'n_steps':
            model.n_steps = new_value
            
        config['ppo'][param] = new_value
        
        print(f"Applied adjustment: {param} = {new_value}")
