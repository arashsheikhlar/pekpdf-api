#!/usr/bin/env python3
"""
Simple test to debug Ollama calling
"""

import requests
import json

def call_ollama_direct(prompt, system_prompt=""):
    """Direct call to Ollama to test"""
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3.1:8b",
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 4096,
                "repeat_penalty": 1.1,
                "seed": -1
            }
        }
        
        print(f"Calling Ollama with prompt: {prompt[:100]}...")
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', 'No response from AI model')
            print(f"SUCCESS: {response_text[:200]}...")
            return response_text
        else:
            print(f"ERROR: Status {response.status_code}")
            return f"Error calling Ollama: {response.status_code}"
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return f"Error connecting to Ollama: {str(e)}"

def test_different_questions():
    """Test with different questions"""
    questions = [
        "What is the capital of France?",
        "Tell me about cats", 
        "What is 2 + 2?",
        "How do airplanes fly?",
        "What is Python programming?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i}: {question} ---")
        response = call_ollama_direct(question)
        print()

if __name__ == "__main__":
    test_different_questions() 