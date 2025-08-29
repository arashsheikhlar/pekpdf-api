# AI Tools Setup Guide for Render

This guide will help you configure the AI tools in PerkPDF to work properly on Render.

## Environment Variables to Set on Render

You need to set these environment variables in your Render service:

### 1. AI Service Selection
```
AI_SERVICE=anthropic
```
This tells the app to use Anthropic instead of Ollama.

### 2. Anthropic API Key
```
ANTHROPIC_API_KEY=your_actual_api_key_here
```
Get this from your Anthropic account at https://console.anthropic.com/

### 3. Anthropic Model (Optional)
```
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```
This is the default, but you can also use:
- `claude-3-5-haiku-20241022` (faster, cheaper)
- `claude-3-5-opus-20241022` (most capable, more expensive)

## How to Set Environment Variables on Render

1. Go to your Render dashboard
2. Select your PerkPDF service
3. Go to "Environment" tab
4. Add each variable:
   - Key: `AI_SERVICE`, Value: `anthropic`
   - Key: `ANTHROPIC_API_KEY`, Value: `your_api_key`
   - Key: `ANTHROPIC_MODEL`, Value: `claude-3-5-sonnet-20241022`

## Testing Your Configuration

After setting the environment variables:

1. Deploy your service
2. Test the AI tools by uploading a PDF and using any AI feature
3. Check the logs for any error messages

## Common Issues and Solutions

### "Invalid API key" error
- Make sure your API key is correct
- Check that your Anthropic account is active
- Verify you have credits in your account

### "Invalid model" error
- Make sure the model name is exactly correct
- Check that you have access to the model in your account
- Try using `claude-3-5-haiku-20241022` as it's usually available

### "Rate limit" error
- You've hit the API rate limits
- Wait a minute and try again
- Consider upgrading your Anthropic plan for higher limits

### "Billing/quota" error
- Check your Anthropic account billing
- Make sure you have sufficient credits
- Verify your payment method is working

## Fallback Options

If Anthropic doesn't work, you can:

1. **Use Ollama** (local AI):
   ```
   AI_SERVICE=ollama
   OLLAMA_BASE_URL=http://your-ollama-instance:11434
   OLLAMA_MODEL=llama3.1:8b
   ```

2. **Use OpenAI** (if you have an OpenAI account):
   ```
   AI_SERVICE=openai
   OPENAI_API_KEY=your_openai_key
   OPENAI_MODEL=gpt-3.5-turbo
   ```

## Testing Locally

You can test your configuration locally by:

1. Creating a `.env` file with your settings
2. Running the test script:
   ```bash
   python test_anthropic_config.py
   ```

This will help identify any configuration issues before deploying to Render. 