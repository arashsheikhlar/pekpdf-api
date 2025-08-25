#!/usr/bin/env python3
"""
Test script to verify Ollama is working and giving different responses
"""

import requests
import json

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"

def test_ollama():
    questions = [
        "What is the capital of France?",
        "What is 2 + 2?", 
        "Tell me about cats",
        "What is the weather like?",
        "How do you make coffee?"
    ]
    
    print("Testing Ollama with different questions...")
    print("=" * 50)
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        
        try:
            url = f"{OLLAMA_BASE_URL}/api/generate"
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": question,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 4096,
                    "repeat_penalty": 1.1,
                    "seed": -1
                }
            }
            
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', 'No response')
                print(f"Answer: {answer[:200]}...")
            else:
                print(f"Error: {response.status_code}")
                
        except Exception as e:
            print(f"Exception: {e}")
    
    print("\n" + "=" * 50)
    print("If all answers are different, Ollama is working correctly.")

if __name__ == "__main__":
    test_ollama() 