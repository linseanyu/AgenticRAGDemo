# AgenticRAGDemo


## HuggingFace Setup
if you encounter "invalid credential header authentication" problem, put this in your code
```python
from huggingface_hub import login
login(token="HUGGINGFACE_TOKEN")
```

## environment variables
environment variables setup needs to be put at the very beginning of the file
```python
import os
os.environ["USER_AGENT"] = "AgenticRAGDemo/1.0"
# Set HUGGINGFACEHUB_API_TOKEN as an environment variable
HUGGINGFACE_TOKEN = "replace_with_your_token"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_TOKEN

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

## About Model
Not every model call handle tool-calling
When init a model in this way
```python
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", 
    task="text-generation",
    max_new_tokens=512,
    do_sample=True,  # Enable sampling
    temperature=0.7,  # Set temperature for more varied outputs
    top_p=0.95,       # Use top_p sampling
)
chat_model = ChatHuggingFace(llm=llm, verbose=False)
```
the model will never trigger tool-calling, it might because HuggingFaceEndpoint will use tool-calling directly and generate the final answer to return. Instead, use the following way to create a chat model
```python
#### Use Local Ollama Model
chat_model = init_chat_model(
    model="llama3.2",  # Model name as specified in Ollama
    model_provider="ollama",  # Use Ollama provider
    base_url="http://localhost:11434",  # Default Ollama server URL
    temperature=0.6,  # Optional: Control randomness
    max_tokens=256  # Optional: Limit response length
)
```


## Questions
* What's the main concept of langchain. How it works?
* What HuggingFaceEmbeddings does behind the scene?
* What does FAISS do ? What's meta data in the context of vectorStore?
* What's HuggingFaceEndpoint and ChatHuggingFace? What's the difference?
* What's MessageState in LangChain ? Why do we need MessageState?
* What does bind_tools does? How do a model know if it should use tool and which tool to use? What's the difference between using or not using a tool in terms of return value? Can model use multiple tools at the same time.
* What do those LangGraph API mean. For exmaple: add_conditional_edges, add_edge, add_node ? 
