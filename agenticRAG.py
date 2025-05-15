import os
import json
os.environ["USER_AGENT"] = "AgenticRAGDemo/1.0"
# Set HUGGINGFACEHUB_API_TOKEN as an environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "replace_with_your_token"

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import convert_to_messages

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
# Set USER_AGENT environment variable to avoid the warning
from huggingface_hub import login
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama

# Flag to control whether to rebuild the index even if it exists
REBUILD_INDEX = False

######### Login HuggingFace #########
login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])

######### Process Documents #########
urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    # "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    # "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

# Define the path for the FAISS index
FAISS_INDEX_PATH = "faiss_index"

# Only process documents if we're rebuilding the index or it doesn't exist
if REBUILD_INDEX or not os.path.exists(FAISS_INDEX_PATH):
    print("Processing documents...")
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(docs_list)
    print(f"Processed {len(doc_splits)} document chunks")
else:
    # Create empty placeholder when using existing index
    doc_splits = []
    print("Skipping document processing since using existing index")

######### Create a retriever tool
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Use the full model name
)

# ######### Create a in-memory vector store
# docs = [WebBaseLoader(url).load() for url in urls]

# # print(docs[0][0].page_content.strip()[:1000])

# docs_list = [item for sublist in docs for item in sublist]

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)

# doc_splits = text_splitter.split_documents(docs_list)
# vector_store = InMemoryVectorStore.from_documents(
#     documents=doc_splits,
#     embedding=embeddings,
# )

# Check if FAISS index already exists
if os.path.exists(FAISS_INDEX_PATH) and not REBUILD_INDEX:
    print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}")
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings,
        allow_dangerous_deserialization=True  # Add this parameter to allow loading the pickle file
    )
else:
    print("Building new FAISS index from documents")
    vector_store = FAISS.from_documents(
        documents=doc_splits,
        embedding=embeddings,
    )
    # Save the index for future use
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")

retriever = vector_store.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "A tool to retrieve documents from the vector store",
)

# print(retriever_tool.invoke("What is reward hacking?"))

# ######### Generate query
# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", 
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=True,  # Enable sampling
#     temperature=0.7,  # Set temperature for more varied outputs
#     top_p=0.95,       # Use top_p sampling
# )
# chat_model = ChatHuggingFace(llm=llm, verbose=False)

chat_model = init_chat_model(
    model="llama3.2",  # Model name as specified in Ollama
    model_provider="ollama",  # Use Ollama provider
    base_url="http://localhost:11434",  # Default Ollama server URL
    temperature=0.6,  # Optional: Control randomness
    max_tokens=256  # Optional: Limit response length
)

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        chat_model.bind_tools([retriever_tool]).invoke(state["messages"])
    )
    return {"messages": [response]}

# # input = {"messages": [{"role": "system", "content": "Use retriever_tool only when it's necessary to answer the question"}, {"role": "user", "content": "Hello, how are you?"}]}
# # generate_query_or_respond(input)["messages"][-1].pretty_print()

# input = {"messages": [{"role": "system", "content": "Use retriever_tool only when it's necessary to answer the question"}, {"role": "user", "content": "What does Lilian Weng say about types of reward hacking?"}]}
# generate_query_or_respond(input)["messages"][-1].pretty_print()

######### Grade documents
GRADE_PROMPT = (
    "You are a grader assessing the relevance of a retrieved document to a user question. "
    "Retrieved document: {context} "
    "User question: {question} "
    "Output EXACTLY the string 'yes' or 'no' to indicate whether the document contains keywords or semantic meaning related to the question. "
    "Do NOT include any additional text, explanations, comments, newlines, or spaces before or after the output. "
    "Strictly output 'yes' or 'no' and nothing else."
)

grader_model = init_chat_model(
    model="llama3.2",  # Model name as specified in Ollama
    model_provider="ollama",  # Use Ollama provider
    base_url="http://localhost:11434",  # Default Ollama server URL
    temperature=0.6,  # Optional: Control randomness
    max_tokens=256  # Optional: Limit response length
)

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model.invoke(
            [   
                {"role": "user", "content": prompt}
            ]
        )
    )

    if response.content.strip().lower() == "yes":
        print("grade_documents: generate_answer")
        return "generate_answer"
    else:
        print("grade_documents: rewrite_question")
        return "rewrite_question"


REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = grader_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}

# input = {
#     "messages": convert_to_messages(
#         [
#             {
#                 "role": "user",
#                 "content": "What does Lilian Weng say about types of reward hacking?",
#             },
#             {
#                 "role": "assistant",
#                 "content": "",
#                 "tool_calls": [
#                     {
#                         "id": "1",
#                         "name": "retrieve_blog_posts",
#                         "args": {"query": "types of reward hacking"},
#                     }
#                 ],
#             },
#             {"role": "tool", "content": "meow", "tool_call_id": "1"},
#         ]
#     )
# }

# response = rewrite_question(input)
# print(response["messages"][-1]["content"])

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = grader_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

# response = generate_answer(input)
# response["messages"][-1].pretty_print()

######### Assemble the graph

workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()

for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            }
        ]
    }
):
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")