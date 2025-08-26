# Production Setup Guide for Perk PDF AI Tools

## Problem
AI tools work locally with Ollama but fail on Render with "AI Chat failed" errors.

## Root Cause
Ollama is not running on Render servers. Ollama requires local installation and cannot run on Render's Python runtime.

## Solutions

### Option 1: Use OpenAI (Recommended for Production)

#### 1. Get OpenAI API Key
- Visit [OpenAI Platform](https://platform.openai.com/api-keys)
- Create a new API key
- Copy the key

#### 2. Configure Environment Variables on Render
In your Render dashboard:
1. Go to your service `simurghpdf-api`
2. Click "Environment"
3. Add these variables:
   ```
   AI_SERVICE=openai
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-3.5-turbo
   ```

#### 3. Deploy
- Push your changes to GitHub
- Render will automatically rebuild and deploy

### Option 2: Use Anthropic Claude (Alternative)

#### 1. Get Anthropic API Key
- Visit [Anthropic Console](https://console.anthropic.com/)
- Create a new API key

#### 2. Configure Environment Variables
   ```
   AI_SERVICE=anthropic
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ANTHROPIC_MODEL=claude-3-haiku-20240307
   ```

### Option 3: Hybrid Approach (Development + Production)

#### Local Development (.env file)
```env
AI_SERVICE=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

#### Production (Render Environment Variables)
```env
AI_SERVICE=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

## Cost Considerations

### OpenAI GPT-3.5-turbo
- **Input**: $0.0015 per 1K tokens
- **Output**: $0.002 per 1K tokens
- **Typical PDF analysis**: $0.01-0.05 per document

### Anthropic Claude Haiku
- **Input**: $0.00025 per 1K tokens  
- **Output**: $0.00125 per 1K tokens
- **Typical PDF analysis**: $0.005-0.02 per document

## Testing

### 1. Test Locally with OpenAI
```bash
export AI_SERVICE=openai
export OPENAI_API_KEY=your_key_here
python app.py
```

### 2. Test on Render
- Deploy with environment variables
- Test AI tools on production site
- Check logs for any errors

## Troubleshooting

### Common Issues:

#### 1. "OpenAI API key not configured"
- Check environment variables on Render
- Ensure `AI_SERVICE=openai` is set

#### 2. "Rate limit exceeded"
- OpenAI has rate limits on free tier
- Consider upgrading or using Anthropic

#### 3. "Model not found"
- Check `OPENAI_MODEL` environment variable
- Use valid model names like `gpt-3.5-turbo`

## Security Notes

- Never commit API keys to Git
- Use Render environment variables
- Monitor API usage and costs
- Consider implementing rate limiting

## Next Steps

1. Choose your AI service (OpenAI recommended)
2. Get API key and configure environment variables
3. Deploy to Render
4. Test AI tools on production
5. Monitor costs and usage 