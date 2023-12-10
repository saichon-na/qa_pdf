import os
import openai

import tiktoken
from llama_index.text_splitter import TokenTextSplitter
from llama_index.response.schema import Response, StreamingResponse
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.callbacks.base import CallbackManager
from llama_index.node_parser import SimpleNodeParser
from llama_index import (
    LLMPredictor,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    PromptHelper,
    load_index_from_storage,
)
from langchain.chat_models import ChatOpenAI
import chainlit as cl


openai.api_key = os.environ.get("OPENAI_API_KEY")

STREAMING = True

## Function to load the index from storage or create a new one
@cl.cache  ## Allow to cache the function
def load_context():
    '''
    try:
        llm_predictor = LLMPredictor(
            llm=ChatOpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo",
                streaming=STREAMING,
            ),
        )
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            chunk_size=512,
            callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
        )
        # Rebuild the storage context
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage",
            callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
        )
        # Load the index
        index = load_index_from_storage(
            storage_context, storage_context=storage_context
        )
    except:

        #from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
        from llama_index import SimpleDirectoryReader

        documents = SimpleDirectoryReader("./pdfs").load_data()
        #index = GPTVectorStoreIndex.from_documents(documents)
        #index.storage_context.persist()
        
        llm_predictor = LLMPredictor(
            llm=ChatOpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo-16k",
                streaming=True,
            ),
        )
        prompt_helper = PromptHelper(
            context_window=4096,
            num_output=256,
            chunk_overlap_ratio=0.1,
            chunk_size_limit=None,
        )
        # embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
        text_splitter = TokenTextSplitter(
        separator=" ", chunk_size=1024, chunk_overlap=20,
        backup_separators=["\n"],
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
        )
        node_parser = SimpleNodeParser()

        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            # embed_model=embed_model, ## (optional)
            #node_parser=node_parser, ## (optional)
            prompt_helper=prompt_helper,
            callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
        )
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        index.storage_context.persist("./storage")
    return index
    '''
    from pathlib import Path
    from llama_index import download_loader

    PDFReader = download_loader("PDFReader")

    loader = PDFReader()
    documents = loader.load_data(file=Path('./pdfs/gemini_1_report.pdf'))
    
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo-16k",
            streaming=True,
        ),
    )
    prompt_helper = PromptHelper(
        context_window=4096,
        num_output=256,
        chunk_overlap_ratio=0.1,
        chunk_size_limit=None,
    )
    # embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
    text_splitter = TokenTextSplitter(
    separator=" ", chunk_size=1024, chunk_overlap=20,
    backup_separators=["\n"],
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    )
    node_parser = SimpleNodeParser()

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        # embed_model=embed_model, ## (optional)
        #node_parser=node_parser, ## (optional)
        prompt_helper=prompt_helper,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )
    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context
    )
    index.storage_context.persist("./storage")

    return index

@cl.on_chat_start
async def factory():
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            streaming=STREAMING,
        ),
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        chunk_size=512,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )

    index = load_context()

    if index is None:
        await cl.Message(author="Assistant", content="index is none").send()
        # Rebuild the storage context
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage",
            callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
        )
        # Load the index
        index = load_index_from_storage(
            storage_context, storage_context=storage_context
        )
    
    else:
        query_engine = index.as_query_engine(
            service_context=service_context,
            streaming=STREAMING,
        )

        cl.user_session.set("query_engine", query_engine)
        
    await cl.Message(author="Assistant", content="Hello ! How may I help you ? ").send()


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    response = await cl.make_async(query_engine.query)(message.content)

    response_message = cl.Message(content="")

    if isinstance(response, Response):
        response_message.content = str(response)
        await response_message.send()
    elif isinstance(response, StreamingResponse):
        gen = response.response_gen
        for token in gen:
            await response_message.stream_token(token=token)

        if response.response_txt:
            response_message.content = response.response_txt

        await response_message.send()