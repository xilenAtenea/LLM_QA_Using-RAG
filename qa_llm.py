from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma


def loaders():
    """
    Load documents using PyPDFLoader for local PDF files and WebBaseLoader for web pages.

    Returns:
        docs (list): A list of documents loaded from the specified sources.
    """
    loaders = [
        PyPDFLoader("info/Código-de-Ética-para-Estudiantes.pdf"),
        PyPDFLoader("info/resolucion_No666.pdf" ),
        PyPDFLoader("info/Resolucion_de_Rectoria_No._7714_reglamento_academico_de_pregrado_Modalidad_Virtual.pdf" ),
        WebBaseLoader("https://www.uao.edu.co/solicitudes-de-supletorios-validaciones-y-habilitaciones/")
        ]

    docs = []

    for loader in loaders:
        docs.extend(loader.load())
    
    return docs


def split_data(docs):
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.

    Args:
        docs (list): A list of documents to split.

    Returns:
        all_splits (list): A list of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter( 
        chunk_size=1500, chunk_overlap=150)

    all_splits = text_splitter.split_documents(docs)

    return all_splits


def create_vector_store(all_splits):
    """
    Create a vector store and retriever from document splits.

    Args:
        all_splits (list): A list of document chunks.

    Returns:
        retriever: A retriever for querying the vector store.
    """
    persist_directory = 'data'

    vectorstore = Chroma.from_documents( 
        documents=all_splits, 
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory=persist_directory
    )

    retriever = vectorstore.as_retriever()

    return retriever


def llm_model():
    """
    Creates and returns an instance of the LLM (Language Model) class.
    
    Returns:
        llm: An instance of the LLM class.
    """
    llm = Ollama(base_url="http://localhost:11434",
                 model="llama3",
                 verbose=False,
                 stop=['<|eot_id|>'],
                 callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                 )
    
    return llm


def create_prompt():
    """
    Creates the prompt template for the chatbot.

    Returns:
        promt: The created prompt template.
    """
    template = """ You are a chatbot providing information to students, collaborators, 
    and professors of Universidad Autónoma de Occidente. Your tone should be professional 
    and informative. Keep responses concise, addressing only what users ask. If you don't 
    know the answer, simply state so. You have to answer in the language used by the user. 
    Use the following context to response the questions.
        
    Context: {context}
    History: {history}

    Question: {question}
    Chatbot:
    """

    prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=template,
        )

    return prompt


def create_memory():
    """
    Creates a memory object to store the conversation history.

    Returns:
        memory: The created memory object.
    """    
    memory = ConversationBufferMemory(
                memory_key="history", 
                return_messages=True,
                input_key="question" 
            )
    
    return memory


def create_retrieval_chain(llm, retriever,prompt, memory):
    """
    Creates a retrieval chain for question answering.

    Args:
        llm (LLM): The LLM model.
        retriever (Retriever): The retriever object.
        prompt (str): The prompt for the retrieval chain.
        memory (str): The memory for the retrieval chain.

    Returns:
        qa_chain (object): The retrieval chain for question answering.
    """
    
    qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=retriever,
                chain_type_kwargs={
                    "prompt": prompt,
                    "memory": memory
                }
            )
    
    return qa_chain    

def pretty_print(answer):
    """
    Prints the chat conversation between the user and the chatbot in a formatted manner.
    
    Args:
        answer (dict): A dictionary containing the query and result of the chat conversation.
        
    Returns:
        None
    """
    print(f"""
      UAO CHATBOT
            
      User: {answer.get("query")} 
      Chat: {answer.get("result")}
            """)

def chat_loop(qa_chain):
    """
    Function to start a chat loop with the user.

    Args:
        qa_chain (object): The question answering chain object.

    Returns:
        None
    """
    print("BIENVENID@ A CHAT UAO")
    while True:
        query = input("¿En qué te puedo ayudar? (Escribe 'salir' para terminar)\n-> ")
        if query.lower() == 'salir':
            print("¡Hasta luego!")
            break
        answer = qa_chain.invoke({"query": query})
        pretty_print(answer)

def main():
    """
    Main function that executes the question answering process.
    """
    docs = loaders()
    all_splits = split_data(docs)
    retriever = create_vector_store(all_splits)
    llm = llm_model()
    prompt = create_prompt()
    memory = create_memory()
    qa_chain = create_retrieval_chain(llm, retriever, prompt, memory)
    chat_loop(qa_chain)
    

if __name__ == "__main__":
    main()

    