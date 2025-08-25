# Ollama Setup Guide for Perk PDF AI Tools

## What is Ollama?
Ollama is a free, open-source tool that allows you to run large language models (LLMs) locally on your machine. It's perfect for AI-powered PDF analysis without any usage limits or costs.

## Installation

### 1. Install Ollama
Visit [ollama.ai](https://ollama.ai) and download the installer for your operating system:
- **Windows**: Download and run the Windows installer
- **macOS**: Download and run the macOS installer
- **Linux**: Run the installation script

### 2. Install the Llama 3.1 Model
After installing Ollama, open a terminal/command prompt and run:
```bash
ollama pull llama3.1:8b
```

This downloads the Llama 3.1 8B model (~4.7GB) which provides excellent performance for document analysis.

### 3. Start Ollama Service
Ollama runs as a local service on port 11434. The service should start automatically after installation.

## Configuration

### Environment Variables (Optional)
Create a `.env` file in your backend directory:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

### Default Settings
- **Base URL**: `http://localhost:11434`
- **Model**: `llama3.1:8b`

## Testing Ollama

### 1. Test the Service
Open a terminal and run:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Hello, how are you?"
}'
```

### 2. Test with Python
```python
import requests

response = requests.post('http://localhost:11434/api/generate', json={
    'model': 'llama3.1:8b',
    'prompt': 'Hello, how are you?'
})

print(response.json()['response'])
```

## Available Models

### Recommended Models for PDF Analysis:
- **llama3.1:8b** - Best balance of performance and speed (~4.7GB)
- **llama3.1:70b** - Highest quality but slower (~40GB)
- **mistral:7b** - Fast and efficient (~4.1GB)
- **codellama:7b** - Great for technical documents (~4.1GB)

### Install Additional Models:
```bash
ollama pull mistral:7b
ollama pull codellama:7b
```

## Performance Tips

### 1. Hardware Requirements
- **Minimum**: 8GB RAM, 4GB free disk space
- **Recommended**: 16GB+ RAM, SSD storage
- **GPU**: Optional but significantly faster (NVIDIA/AMD)

### 2. Model Selection
- **8B models**: Fast, good quality, lower memory usage
- **70B models**: Highest quality, slower, higher memory usage

### 3. Response Time
- **8B models**: 2-5 seconds per response
- **70B models**: 5-15 seconds per response

## Troubleshooting

### Common Issues:

#### 1. Ollama Service Not Running
```bash
# Windows: Check Services app
# macOS/Linux: Check if process is running
ps aux | grep ollama

# Restart service
ollama serve
```

#### 2. Model Not Found
```bash
# List installed models
ollama list

# Reinstall model
ollama pull llama3.1:8b
```

#### 3. Port Already in Use
```bash
# Check what's using port 11434
netstat -an | grep 11434

# Kill process or change port in .env
OLLAMA_BASE_URL=http://localhost:11435
```

#### 4. Memory Issues
- Use smaller models (8B instead of 70B)
- Close other applications
- Restart Ollama service

## Security & Privacy

### Benefits:
- **100% Local**: No data leaves your machine
- **No Rate Limits**: Unlimited usage
- **No Costs**: Completely free
- **Privacy**: Your PDFs never leave your system

### Considerations:
- Models are downloaded to your machine
- Requires local storage and processing power
- No cloud backup of models

## Integration with Perk PDF

The AI Tools in Perk PDF now use Ollama for:
- **Chat Mode**: Interactive conversations about PDF content
- **Explain Mode**: Comprehensive document explanations
- **Ask Mode**: Specific question answering
- **Summarize Mode**: Document summarization

All responses are generated locally using your installed Ollama models.

## Support

- **Ollama Documentation**: [ollama.ai/docs](https://ollama.ai/docs)
- **Community**: [GitHub Discussions](https://github.com/ollama/ollama/discussions)
- **Perk PDF Issues**: Report bugs in the main repository

## Next Steps

1. Install Ollama following the steps above
2. Pull the Llama 3.1 8B model
3. Start your Perk PDF backend
4. Test the AI Tools with a PDF upload
5. Enjoy unlimited, free AI-powered PDF analysis! 