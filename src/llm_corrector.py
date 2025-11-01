import os
import json
import pandas as pd
from pathlib import Path
from config import (
    LLM_TEMPERATURE, 
    DEFAULT_LLM_PROVIDER, 
    DEFAULT_MODEL,
    PROMPTS_DIR
)


class LLMCorrector:
    def __init__(self, provider=None, model=None, api_key=None):
        """Initialize LLM corrector with specified provider"""
        self.provider = provider or os.getenv('LLM_PROVIDER', DEFAULT_LLM_PROVIDER)
        self.model = model or os.getenv('LLM_MODEL', DEFAULT_MODEL.get(self.provider))
        self.api_key = api_key
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
        
        # Initialize client
        self.client = self._initialize_client()
    
    def _load_system_prompt(self):
        """Load system prompt from file"""
        prompt_file = PROMPTS_DIR / "system_prompt.txt"
        if prompt_file.exists():
            with open(prompt_file, 'r') as f:
                return f.read()
        else:
            raise FileNotFoundError(f"System prompt not found at {prompt_file}")
    
    def _initialize_client(self):
        """Initialize LLM client based on provider"""
        if self.provider == 'openai':
            import openai
            api_key = self.api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found")
            return openai.OpenAI(api_key=api_key)
        
        elif self.provider == 'anthropic':
            import anthropic
            api_key = self.api_key or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key not found")
            return anthropic.Anthropic(api_key=api_key)
        
        elif self.provider == 'google':
            import google.generativeai as genai
            api_key = self.api_key or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Google API key not found")
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.model)
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _call_llm(self, user_message):
        """Call LLM API with user message"""
        try:
            if self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=LLM_TEMPERATURE
                )
                return response.choices[0].message.content
            
            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self.system_prompt,
                    messages=[
                        {"role": "user", "content": user_message}
                    ],
                    temperature=LLM_TEMPERATURE
                )
                return response.content[0].text
            
            elif self.provider == 'google':
                response = self.client.generate_content(
                    f"{self.system_prompt}\n\nUser: {user_message}"
                )
                return response.text
            
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")
    
    def correct_timetable_entry(self, entry_data):
        """Correct a single timetable entry using LLM"""
        user_message = f"Please correct this timetable entry:\n\n{json.dumps(entry_data, indent=2)}"
        
        response = self._call_llm(user_message)
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith('```'):
                # Remove markdown code blocks
                lines = response.split('\n')
                response = '\n'.join([l for l in lines if not l.strip().startswith('```')])
            
            corrected_data = json.loads(response)
            return corrected_data
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response: {response}")
            return entry_data  # Return original if parsing fails
    
    def correct_dataframe(self, df):
        """Correct entire DataFrame using LLM"""
        if df.empty:
            return df
        
        corrected_rows = []
        
        for idx, row in df.iterrows():
            # Convert row to dict
            entry_data = row.to_dict()
            
            # Skip empty rows
            if all(str(v).strip() == '' or pd.isna(v) for v in entry_data.values()):
                corrected_rows.append(entry_data)
                continue
            
            # Correct using LLM
            try:
                corrected_entry = self.correct_timetable_entry(entry_data)
                corrected_rows.append(corrected_entry)
            except Exception as e:
                print(f"Error correcting row {idx}: {e}")
                corrected_rows.append(entry_data)
        
        corrected_df = pd.DataFrame(corrected_rows)
        return corrected_df
    
    def batch_correct_dataframe(self, df, batch_size=10):
        """Correct DataFrame in batches for efficiency"""
        if df.empty:
            return df
        
        corrected_rows = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_data = batch.to_dict('records')
            
            # Create batch correction request
            user_message = f"""Please correct these timetable entries. Return a JSON array with all corrected entries:

{json.dumps(batch_data, indent=2)}"""
            
            try:
                response = self._call_llm(user_message)
                
                # Parse response
                response = response.strip()
                if response.startswith('```'):
                    lines = response.split('\n')
                    response = '\n'.join([l for l in lines if not l.strip().startswith('```')])
                
                corrected_batch = json.loads(response)
                corrected_rows.extend(corrected_batch)
            except Exception as e:
                print(f"Error correcting batch {i//batch_size}: {e}")
                corrected_rows.extend(batch_data)
        
        corrected_df = pd.DataFrame(corrected_rows)
        return corrected_df
