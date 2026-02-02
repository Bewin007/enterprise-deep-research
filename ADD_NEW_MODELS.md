# How to Add New Models and Providers to Enterprise Deep Research

This guide explains how to add new AI models and providers to your Enterprise Deep Research system. It's written for non-technical users who want to expand their model options.

## Table of Contents

1. [Quick Overview](#quick-overview)
2. [Adding a New Ollama Model](#adding-a-new-ollama-model)
3. [Adding a New Cloud Provider](#adding-a-new-cloud-provider)
4. [Adding a New Self-Hosted Provider](#adding-a-new-self-hosted-provider)
5. [Troubleshooting](#troubleshooting)

## Quick Overview

To add a new model, you need to make changes in **two places**:
1. **Backend** (Python files) - Tells the system how to connect to the model
2. **Frontend** (JavaScript files) - Shows the model in the dropdown menu

**Important**: After making changes, you need to rebuild the frontend and restart the backend.

## Adding a New Ollama Model

This is the **easiest** way to add new models since Ollama handles everything locally.

### Step 1: Download the Model
Open your terminal and download the model:
```bash
ollama pull llama3.1:7b
# or
ollama pull mistral:latest
# or  
ollama pull codellama:7b
```

### Step 2: Add to Frontend Dropdown

**File to Edit**: `ai-research-assistant/src/components/InitialScreen.js`

**What to Change**: Find the `MODEL_OPTIONS` array (around line 7) and add your new model:

```javascript
const MODEL_OPTIONS = [
  // Existing models...
  {
    key: "ollama-mistral",                    // Unique identifier
    label: "mistral:latest (Local)",         // What users see
    model: "mistral:latest",                 // Exact model name from ollama
    provider: "ollama",                      // Always "ollama" for Ollama models
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="#4F46E5" strokeWidth="2" fill="#4F46E5" fillOpacity="0.2"/>
        <path d="M2 17L12 22L22 17" stroke="#4F46E5" strokeWidth="2"/>
        <path d="M2 12L12 17L22 12" stroke="#4F46E5" strokeWidth="2"/>
      </svg>
    ),
  },
```

**Explanation**:
- `key`: Must be unique for each model
- `label`: What users see in the dropdown (add "(Local)" to show it's self-hosted)
- `model`: Must exactly match what you typed in the `ollama pull` command
- `provider`: Always use `"ollama"`
- `icon`: You can copy the existing icon or create your own

### Step 3: Add to Backend Configuration

**File to Edit**: `llm_clients.py`

**What to Change**: Find the `"ollama"` section in `MODEL_CONFIGS` (around line 117) and add your model:

```python
"ollama": {
    "available_models": [
        "qwen2.5:7b",
        "llama3.1:8b", 
        "mistral:latest",          # Add your new model here
        "codellama:7b",            # And here if you have multiple
        # ... other models
    ],
    "default_model": "qwen2.5:7b",  # You can change this to your preferred default
    "requires_api_key": None,
},
```

### Step 4: Rebuild and Test

1. **Rebuild the frontend**:
   ```bash
   cd ai-research-assistant
   npm run build
   ```

2. **The backend should automatically restart** (if you used `--reload`)

3. **Test**: Open http://localhost:8000 and check if your new model appears in the dropdown

## Adding a New Cloud Provider

### Step 1: Get API Credentials

First, sign up for the service and get your API key:
- **OpenRouter**: Get API key from https://openrouter.ai/
- **Perplexity**: Get API key from https://www.perplexity.ai/
- **Cohere**: Get API key from https://cohere.ai/

### Step 2: Add to Environment Variables

**File to Edit**: `.env`

Add your new API key:
```env
# Add your new provider API key
OPENROUTER_API_KEY=your_api_key_here
PERPLEXITY_API_KEY=your_api_key_here
```

### Step 3: Add Backend Support

**File to Edit**: `llm_clients.py`

**Step 3a**: Add API key variable (around line 25):
```python
# Add after existing API keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
```

**Step 3b**: Add to MODEL_CONFIGS (around line 115):
```python
# Add your new provider
"openrouter": {
    "available_models": [
        "meta-llama/llama-3.1-8b-instruct",
        "anthropic/claude-3-haiku",
        # Add more models as needed
    ],
    "default_model": "meta-llama/llama-3.1-8b-instruct",
    "requires_api_key": OPENROUTER_API_KEY,
},
```

**Step 3c**: Add client creation code (around line 1188):
```python
# Add this in the get_llm_client function, before the final else
elif provider == "openrouter":
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set in environment")
    if not model_name:
        model_name = MODEL_CONFIGS["openrouter"]["default_model"]
    print(f"Using OpenRouter for {model_name}")
    
    # Use OpenAI-compatible client with custom base URL
    return ChatOpenAI(
        model_name=model_name,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",  # Provider's API endpoint
        max_tokens=4000,
    )
```

### Step 4: Add to Frontend

**File to Edit**: `ai-research-assistant/src/components/InitialScreen.js`

Add to the `MODEL_OPTIONS` array:
```javascript
{
  key: "openrouter-llama",
  label: "Llama 3.1 8B (OpenRouter)", 
  model: "meta-llama/llama-3.1-8b-instruct",
  provider: "openrouter",              // Must match what you added in backend
  icon: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      {/* Your custom icon here */}
      <circle cx="12" cy="12" r="10" stroke="#FF6B35" fill="#FF6B35" fillOpacity="0.1"/>
    </svg>
  ),
},
```

### Step 5: Update Frontend Default

**File to Edit**: `ai-research-assistant/src/App.js`

If you want your new provider as default, change lines 13-14:
```javascript
const [modelProvider, setModelProvider] = useState('openrouter'); // Your new provider
const [modelName, setModelName] = useState('meta-llama/llama-3.1-8b-instruct'); // Your new model
```

## Adding a New Self-Hosted Provider

### Step 1: Create Client Class

**File to Edit**: `llm_clients_selfhosted.py`

Add a new client class:
```python
class MyCustomClient:
    """
    Client for My Custom AI Server.
    """
    
    def __init__(
        self,
        model_name: str = "my-model",
        base_url: str = None,
        api_key: str = None,
        timeout: int = 120,
        max_tokens: int = 4096,
    ):
        """Initialize the custom client."""
        self.model_name = model_name
        self.base_url = base_url or os.getenv("MYCUSTOM_BASE_URL", "http://localhost:8080")
        self.api_key = api_key or os.getenv("MYCUSTOM_API_KEY", "")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.model = model_name
    
    def invoke(self, messages, config=None, stream=False):
        """Send request to your custom AI server."""
        # Convert LangChain messages to your API format
        custom_messages = []
        for msg in messages:
            role = "user"
            if hasattr(msg, 'type') and msg.type == "system":
                role = "system"
            custom_messages.append({"role": role, "content": msg.content})
        
        # Make API request to your server
        import requests
        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "model": self.model_name,
                "messages": custom_messages,
                "max_tokens": self.max_tokens
            },
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        # Extract response text (adjust based on your API format)
        return result["choices"][0]["message"]["content"]
```

### Step 2: Register the New Provider

**File to Edit**: `llm_clients_selfhosted.py`

Add to `SELFHOSTED_MODEL_CONFIGS`:
```python
SELFHOSTED_MODEL_CONFIGS = {
    # ... existing providers
    "mycustom": {
        "available_models": [
            "my-model-v1",
            "my-model-v2",
        ],
        "default_model": "my-model-v1",
        "client_class": MyCustomClient,  # Your class name
    },
}
```

### Step 3: Add Environment Variables

**File to Edit**: `.env`

```env
# Your custom provider settings
MYCUSTOM_BASE_URL=http://localhost:8080
MYCUSTOM_API_KEY=your_api_key_if_needed
```

### Step 4: Add to Frontend

Follow the same steps as adding a cloud provider, but use `"mycustom"` as the provider name.

## Troubleshooting

### Problem: Model doesn't appear in dropdown
**Solution**: 
1. Check if you added it to both `InitialScreen.js` and `llm_clients.py`
2. Make sure you rebuilt the frontend: `npm run build`
3. Check for typos in the model name

### Problem: "Model not found" error
**Solution**:
1. For Ollama: Run `ollama list` to verify the model is downloaded
2. For cloud providers: Check if your API key is correct in `.env`
3. Check if the model name exactly matches what the provider expects

### Problem: "Provider not supported" error
**Solution**:
1. Make sure you added the provider to `get_llm_client` function
2. Check that the provider name in frontend matches backend
3. Restart the backend after changes

### Problem: API key errors
**Solution**:
1. Check if the API key is properly set in `.env` file
2. Make sure there are no extra spaces or quotes around the API key
3. Verify the API key is valid by testing it directly with the provider

### Problem: Connection timeout
**Solution**:
1. For local models: Check if the service is running (`ollama list`, etc.)
2. For cloud providers: Check your internet connection
3. Increase timeout in the client configuration

## Quick Reference: Files to Edit

| Task | Backend Files | Frontend Files |
|------|--------------|----------------|
| Add Ollama model | `llm_clients.py` | `InitialScreen.js` |
| Add cloud provider | `.env`, `llm_clients.py` | `InitialScreen.js`, `App.js` |
| Add self-hosted provider | `.env`, `llm_clients_selfhosted.py` | `InitialScreen.js`, `App.js` |

## Commands to Remember

```bash
# Download Ollama model
ollama pull model-name:tag

# Rebuild frontend
cd ai-research-assistant && npm run build

# Restart backend (if not using --reload)
cd .. && source venv/bin/activate && python -m uvicorn app:app --host 0.0.0.0 --port 8000

# Test model works
ollama run model-name "test message"

# Check logs
# Backend logs are shown in terminal
# Frontend errors: Check browser developer tools (F12)
```

## Tips for Success

1. **Always make small changes**: Add one model at a time and test
2. **Check names carefully**: Model names are case-sensitive
3. **Test with simple queries first**: Use "Hello" or "What is AI?" to verify
4. **Keep backups**: Copy files before editing them
5. **Read error messages**: They usually tell you exactly what's wrong
6. **Use exact model names**: Copy-paste from provider documentation

Need help? Check the error messages in your browser's developer tools (press F12) or the backend terminal output for specific guidance!