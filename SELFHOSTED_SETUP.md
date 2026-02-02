# Self-Hosted LLM Setup Guide for Enterprise Deep Research

This guide provides comprehensive instructions for running Enterprise Deep Research with self-hosted Large Language Models (LLMs) using various inference engines.

## Table of Contents

1. [Quick Start with Docker](#quick-start-with-docker)
2. [Supported Inference Engines](#supported-inference-engines)
3. [Configuration Options](#configuration-options)
4. [Model Selection Guide](#model-selection-guide)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Configurations](#advanced-configurations)

## Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (optional but recommended)
- At least 16GB RAM (32GB recommended for larger models)
- 50GB+ free disk space

### Basic Setup with Ollama

1. **Clone the repository:**
```bash
git clone https://github.com/SalesforceAIResearch/enterprise-deep-research.git
cd enterprise-deep-research
```

2. **Create environment file:**
```bash
cp .env.sample .env
```

3. **Configure for Ollama:**
Edit `.env` file:
```env
# Required API keys
TAVILY_API_KEY=your_tavily_key_here
E2B_API_KEY=your_e2b_key_here

# LLM Configuration for Ollama
LLM_PROVIDER=ollama
LLM_MODEL=llama3.3:latest
OLLAMA_BASE_URL=http://ollama:11434

# Activity generation with Ollama
ACTIVITY_LLM_PROVIDER=ollama
ACTIVITY_LLM_MODEL=llama3.3:latest

# Optional settings
MAX_WEB_RESEARCH_LOOPS=10
ENABLE_ACTIVITY_GENERATION=true
```

4. **Start the services:**
```bash
docker-compose up -d
```

This will:
- Start Ollama service
- Automatically pull the llama3.3 model
- Start the EDR application
- Make it available at http://localhost:8000

5. **Verify installation:**
```bash
# Check if services are running
docker-compose ps

# View logs
docker-compose logs -f

# Test the API
curl http://localhost:8000/health
```

## Supported Inference Engines

### 1. Ollama (Recommended for Beginners)

**Pros:**
- Easy setup and management
- Automatic model downloads
- Good performance
- Wide model support

**Setup:**
```yaml
# docker-compose.yml (already included)
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
```

**Available Models:**
- `llama3.3:latest` - Latest Llama 3.3 model (recommended)
- `llama3.2:latest` - Llama 3.2 model
- `llama3.1:8b` - Llama 3.1 8B parameters
- `mistral:latest` - Mistral 7B
- `codellama:latest` - Code-specialized Llama
- `phi3:latest` - Microsoft Phi-3
- `gemma2:latest` - Google Gemma 2
- `qwen2.5:latest` - Alibaba Qwen 2.5

**Pull additional models:**
```bash
docker exec edr-ollama ollama pull mistral:latest
docker exec edr-ollama ollama pull codellama:latest
```

### 2. LocalAI (Multi-Backend Support)

**Pros:**
- Supports multiple backends (llama.cpp, GPT4All, etc.)
- OpenAI-compatible API
- Supports GGUF/GGML models
- Can run on CPU

**Enable LocalAI:**
```bash
docker-compose --profile localai up -d
```

**Configuration in .env:**
```env
LLM_PROVIDER=localai
LLM_MODEL=llama-3-8b
LOCALAI_BASE_URL=http://localai:8080
```

**Install models:**
```bash
# Download model to LocalAI
docker exec -it edr-localai bash
cd /models
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf
```

### 3. vLLM (High Performance)

**Pros:**
- Optimized for throughput
- PagedAttention for efficient memory usage
- Supports tensor parallelism
- Best for production workloads

**Requirements:**
- NVIDIA GPU with 16GB+ VRAM
- CUDA 11.8+

**Enable vLLM:**
```bash
docker-compose --profile vllm up -d
```

**Configuration in .env:**
```env
LLM_PROVIDER=vllm
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
VLLM_BASE_URL=http://vllm:8000
```

### 4. llama.cpp Server (Low Resource)

**Pros:**
- Runs on CPU
- Quantized models (4-bit, 5-bit)
- Low memory usage
- Good for edge deployment

**Setup without Docker:**
```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Download a quantized model
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

# Start server
./server -m llama-2-7b.Q4_K_M.gguf -c 2048 --host 0.0.0.0 --port 8080
```

**Configuration in .env:**
```env
LLM_PROVIDER=openai-compatible
LLM_MODEL=llama-2-7b
OPENAI_COMPATIBLE_BASE_URL=http://localhost:8080
```

### 5. Text Generation WebUI

**Pros:**
- Web interface for model management
- Supports multiple loaders
- Extension system
- Good for experimentation

**Setup:**
```bash
# Clone and setup
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
bash start_linux.sh

# Enable API
# In UI: Settings > API > Enable API
```

**Configuration in .env:**
```env
LLM_PROVIDER=openai-compatible
LLM_MODEL=your-loaded-model
OPENAI_COMPATIBLE_BASE_URL=http://localhost:5000
```

## Configuration Options

### Environment Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `LLM_PROVIDER` | Inference engine to use | `openai` | `ollama`, `localai`, `vllm`, `openai-compatible` |
| `LLM_MODEL` | Model name/path | Provider default | See provider documentation |
| `OLLAMA_BASE_URL` | Ollama API endpoint | `http://localhost:11434` | Any URL |
| `LOCALAI_BASE_URL` | LocalAI API endpoint | `http://localhost:8080` | Any URL |
| `VLLM_BASE_URL` | vLLM API endpoint | `http://localhost:8001` | Any URL |
| `OPENAI_COMPATIBLE_BASE_URL` | Generic endpoint | `http://localhost:8000` | Any URL |
| `MAX_TOKENS` | Max generation tokens | `4096` | 512-32768 |

### Docker Compose Profiles

Use profiles to enable optional services:

```bash
# Basic setup (Ollama only)
docker-compose up -d

# With LocalAI
docker-compose --profile localai up -d

# With vLLM
docker-compose --profile vllm up -d

# Multiple profiles
docker-compose --profile localai --profile vllm up -d
```

## Model Selection Guide

### By Use Case

| Use Case | Recommended Model | Provider | Requirements |
|----------|------------------|----------|--------------|
| General Research | Llama 3.3 | Ollama | 16GB RAM |
| Code Generation | CodeLlama | Ollama | 16GB RAM |
| Fast Responses | Phi-3 | Ollama | 8GB RAM |
| Low Resource | Llama 2 7B Q4 | llama.cpp | 6GB RAM |
| Production | Llama 3.1 8B | vLLM | GPU 16GB VRAM |

### By Hardware

**CPU Only (8-16GB RAM):**
- Phi-3 (Ollama)
- Llama 2 7B Q4 (llama.cpp)
- Mistral 7B Q4 (LocalAI)

**GPU 8GB VRAM:**
- Llama 3.1 7B (vLLM)
- Mistral 7B (Ollama)
- CodeLlama 7B (Ollama)

**GPU 16GB+ VRAM:**
- Llama 3.3 (Ollama)
- Llama 3.1 70B Q4 (vLLM)
- Mixtral 8x7B (vLLM)

## Performance Optimization

### 1. GPU Acceleration

**Enable GPU for Ollama:**
```yaml
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### 2. Model Quantization

**Use quantized models for better performance:**
```bash
# Ollama - automatically uses optimal quantization
docker exec edr-ollama ollama pull llama3.3:8b-q4_0

# LocalAI - use GGUF models
wget https://huggingface.co/TheBloke/Llama-2-13B-GGUF/resolve/main/llama-2-13b.Q4_K_M.gguf
```

### 3. Batch Processing

Configure batch size in vLLM:
```yaml
environment:
  - MAX_BATCH_SIZE=32
  - MAX_NUM_SEQS=256
```

### 4. Memory Management

**Ollama memory settings:**
```bash
# Set GPU memory limit
export OLLAMA_GPU_MEMORY=8192

# Number of parallel requests
export OLLAMA_NUM_PARALLEL=2
```

**vLLM memory settings:**
```yaml
environment:
  - GPU_MEMORY_UTILIZATION=0.95
  - SWAP_SPACE=16
```

## Troubleshooting

### Common Issues

**1. Ollama connection refused:**
```bash
# Check if Ollama is running
docker logs edr-ollama

# Restart Ollama
docker-compose restart ollama

# Test connection
curl http://localhost:11434/api/version
```

**2. Out of memory errors:**
```bash
# Use smaller model
LLM_MODEL=phi3:mini

# Or use quantized version
LLM_MODEL=llama3.1:7b-q4_0

# Reduce context size
export OLLAMA_NUM_CTX=2048
```

**3. Slow generation:**
```bash
# Check GPU usage
nvidia-smi

# Use GPU if available
docker-compose down
docker-compose up -d  # Will use GPU if configured

# Use smaller model or quantization
```

**4. Model not found:**
```bash
# List available models
docker exec edr-ollama ollama list

# Pull required model
docker exec edr-ollama ollama pull llama3.3:latest
```

### Debug Commands

```bash
# View all logs
docker-compose logs -f

# Test Ollama API
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.3:latest",
  "prompt": "Hello world"
}'

# Check resource usage
docker stats

# Enter container for debugging
docker exec -it edr-app bash
```

## Advanced Configurations

### 1. Multi-Model Setup

Run different models for different tasks:

```env
# Main research model
LLM_PROVIDER=ollama
LLM_MODEL=llama3.3:latest

# Fast model for activities
ACTIVITY_LLM_PROVIDER=ollama
ACTIVITY_LLM_MODEL=phi3:mini
```

### 2. Load Balancing

Use multiple inference servers:

```yaml
services:
  ollama1:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
  
  ollama2:
    image: ollama/ollama:latest
    ports:
      - "11435:11434"
  
  nginx:
    image: nginx
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "11436:80"
```

### 3. Custom Models

**Load custom GGUF model in Ollama:**
```bash
# Create Modelfile
cat > Modelfile << EOF
FROM ./my-custom-model.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# Create model
docker exec -it edr-ollama ollama create my-model -f Modelfile
```

### 4. Hybrid Setup

Use cloud models for complex tasks, local for simple:

```python
# In llm_clients.py
def get_hybrid_client(task_complexity):
    if task_complexity == "high":
        return get_llm_client("openai", "gpt-4")
    else:
        return get_llm_client("ollama", "llama3.3:latest")
```

### 5. Monitoring and Metrics

**Add Prometheus monitoring:**
```yaml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

## API Integration Examples

### Python Client

```python
import requests

# Configure for self-hosted
api_url = "http://localhost:8000"
headers = {"Content-Type": "application/json"}

# Start research
response = requests.post(
    f"{api_url}/research",
    json={
        "query": "Latest developments in AI",
        "max_loops": 5
    },
    headers=headers
)

print(response.json())
```

### JavaScript Client

```javascript
const apiUrl = 'http://localhost:8000';

async function startResearch(query) {
    const response = await fetch(`${apiUrl}/research`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query: query,
            max_loops: 5
        })
    });
    
    return response.json();
}

startResearch('Latest AI developments').then(console.log);
```

## Security Considerations

1. **Network Isolation:** Keep inference services on internal network
2. **API Keys:** Even for self-hosted, implement API key authentication
3. **Rate Limiting:** Implement rate limiting for production
4. **Model Validation:** Verify model checksums before deployment
5. **Resource Limits:** Set Docker resource limits to prevent DoS

```yaml
services:
  ollama:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
```

## Migration from Cloud Providers

### Step-by-Step Migration

1. **Test locally first:**
```bash
# Test with small model
LLM_PROVIDER=ollama
LLM_MODEL=phi3:mini
```

2. **Compare outputs:**
```python
# Compare cloud vs local
cloud_result = get_llm_client("openai", "gpt-4").invoke(prompt)
local_result = get_llm_client("ollama", "llama3.3").invoke(prompt)
```

3. **Gradual migration:**
- Start with non-critical tasks
- Monitor performance and accuracy
- Gradually increase usage

4. **Fallback strategy:**
```python
try:
    result = get_llm_client("ollama", "llama3.3").invoke(prompt)
except Exception as e:
    # Fallback to cloud
    result = get_llm_client("openai", "gpt-4").invoke(prompt)
```

## Contributing

To add support for a new inference engine:

1. Implement client in `llm_clients_selfhosted.py`
2. Add to `SELFHOSTED_MODEL_CONFIGS`
3. Update Docker Compose with service definition
4. Add documentation here
5. Submit PR with tests

## Support

- **GitHub Issues:** [Report issues](https://github.com/SalesforceAIResearch/enterprise-deep-research/issues)
- **Documentation:** [Main docs](README.md)
- **Community:** Join discussions in issues

## License

This setup guide and the self-hosted integration code are provided under the same Apache 2.0 license as the main project.