#!/usr/bin/env python3
"""
Startup script for Perk PDF Backend with Ollama integration
"""

import requests
import time
import sys
import os
from app import app

def check_ollama():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("✅ Ollama service is running!")
            if models:
                print(f"📚 Available models: {', '.join([m['name'] for m in models])}")
            else:
                print("⚠️  No models installed. Run 'ollama pull llama3.1:8b' to install a model.")
            return True
        else:
            print(f"❌ Ollama service responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama service is not running!")
        print("💡 Please install and start Ollama:")
        print("   1. Visit https://ollama.ai")
        print("   2. Download and install Ollama")
        print("   3. Run 'ollama pull llama3.1:8b'")
        print("   4. Start Ollama service")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def main():
    print("🚀 Starting Perk PDF Backend...")
    print("=" * 50)
    
    # Check Ollama status
    ollama_ok = check_ollama()
    
    if not ollama_ok:
        print("\n⚠️  AI Tools will not work without Ollama!")
        print("   You can still use other PDF tools (merge, compress, convert, etc.)")
        print("   Continue anyway? (y/N): ", end="")
        
        try:
            response = input().lower().strip()
            if response not in ['y', 'yes']:
                print("❌ Exiting. Please set up Ollama first.")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\n❌ Exiting.")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🌐 Starting Flask server...")
    print("📖 API Documentation: http://localhost:8000/health")
    print("🔗 Frontend should connect to: http://localhost:8000")
    print("=" * 50)
    
    try:
        app.run(host="0.0.0.0", port=8000, debug=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 