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


## Questions
* What's the main concept of langchain. How it works?
* What HuggingFaceEmbeddings does behind the scene?
* What does FAISS do ? What's meta data in the context of vectorStore?
* What's HuggingFaceEndpoint and ChatHuggingFace? What's the difference?
* What's MessageState in LangChain ? Why do we need MessageState?
* What does bind_tools does? How do a model know if it should use tool and which tool to use? What's the difference between using or not using a tool in terms of return value? Can model use multiple tools at the same time.
* What do those LangGraph API mean. For exmaple: add_conditional_edges, add_edge, add_node ? 
