import os
import openai
import chainlit as cl
import PyPDF2

from typing import List
from pathlib import Path

#from llama_index import download_loader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import (
    PyMuPDFLoader,
)
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document,StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory


from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

embeddings_model = OpenAIEmbeddings()

pdf_path = Path(os.environ["PDF_STORAGE_PATH"])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Save upload pdf
def save_uploaded_file(uploaded_file, folder, filename):
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Path where the file will be saved
    file_path = os.path.join(folder, filename)

    if not os.path.exists(file_path):
        # Save the file
        with open(file_path, 'wb') as f:
            f.write(uploaded_file)
        f.close()

        msg = cl.Message(
            content=f"File `{file_path}` has been created.", disable_human_feedback=True
        )
    else:
        msg = cl.Message(
            content=f"`{file_path}` already exists, skip saving.", disable_human_feedback=True
        )
    
    return msg

# pdf processing
def process_pdfs(pdf_storage_path: str):
    pdf_directory = Path(pdf_storage_path)
    docs = []  # type: List[Document]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for pdf_path in pdf_directory.glob("*.pdf"):
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()
        docs += text_splitter.split_documents(documents)

    doc_search = Chroma.from_documents(docs, embeddings_model)

    namespace = "chromadb/my_documents"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    index_result = index(
        docs,
        record_manager,
        doc_search,
        cleanup="incremental",
        source_id_key="source",
    )

    return doc_search

doc_search = process_pdfs(pdf_path)
model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

@cl.on_chat_start
async def on_chat_start():
    template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    retriever = doc_search.as_retriever()

    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    cl.user_session.set("runnable", runnable)

    '''
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["application/pdf"],
            max_size_mb=200,
            timeout=180,
        ).send()

    file = files[0]

    print(dir(file))

    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()

    # Read the PDF file
    pdf_stream = BytesIO(file.content)
    pdf = PyPDF2.PdfReader(pdf_stream)

    # Get the binary data from the BytesIO object
    pdf_bytes = pdf_stream.getvalue()

    await save_uploaded_file(pdf_bytes, pdf_path, file.name).send()

    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    msg = cl.Message(
        content=f"Processing `{pdf_text}`...", disable_human_feedback=True
    )
    await msg.send()

    # Split the text into chunks
    texts = text_splitter.split_text(pdf_text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    
    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)
    '''

async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="describe documents")
    await msg.send()

    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.update()
    '''
    async for chunk in runnable.astream(
        {"question": message.content, "chat_history": history.messages},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await loader_msg.stream_token(chunk.content)
    '''
'''
@cl.on_message
async def main(message: cl.Message):
    #chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    chain = cl.user_session.get("runnable")
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
'''