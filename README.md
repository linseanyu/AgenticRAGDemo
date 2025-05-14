# AgenticRAGDemo


### HuggingFace Setup
if you encounter "invalid credential header authentication" problem, put this in your code
```python
from huggingface_hub import login
login(token="HUGGINGFACE_TOKEN")
```

#### environment variables
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