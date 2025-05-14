# AgenticRAGDemo


## HuggingFace Setup
if you encounter "invalid credential header authentication" problem, put this in your code
```python
from huggingface_hub import login
login(token="HUGGINGFACE_TOKEN")
```