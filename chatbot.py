import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import time
import base64
import uuid
import tempfile
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {} # Save the cache of the uploaded PDF file here so that it can remember.

session_id = st.session_state.id
client = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None

def display_pdf(file):
    ## Opening file from file path

    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    ## Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    ## Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


with st.sidebar:
    # “st” is a package related to UI/UX website configuration with Streamlit. 
    # It is used here to configure the sidebar.

    st.header(f"Add your documents!")
    # A pop-up UX for file upload will automatically appear.
    
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")
    # When a file is uploaded, it is stored in the uploaded_file variable.


    if uploaded_file:
        print(uploaded_file)
        try:
            file_key = f"{session_id}-{uploaded_file.name}"
            # Create a file_key to save the file_cache.

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                print("file path:", file_path)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                    # Function to save the "uploaded file" to a temporary file, open it, and enable indexing.
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")
                # Informing the user that indexing is in progress.

                if file_key not in st.session_state.get('file_cache', {}):

                    if os.path.exists(temp_dir):
                            print("temp_dir:", temp_dir)
                            loader = PyPDFLoader(
                                # Loads the file during preprocessing. "PyPDFLoader" loads the content of the file stored in the temporary directory.
                                # PyPDFLoader is one of the most commonly used examples in Langchain. It handles Korean text processing well and is efficient.
                                # If the document has 10 pages, the content will be split into chunks, with each page stored as a separate chunk: Pages 0, 1, 2, ..., as many as the number of pages.
                                file_path
                            )
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    
                    pages = loader.load_and_split()
                    # "load_and_split" function is used to split the file, and the split content is stored in pages.
                    # Since we are using only one page, there's no need to split it further.

                    vectorstore = Chroma.from_documents(pages, UpstageEmbeddings(model="solar-embedding-1-large"))
                    # The split content is stored in pages, and it needs to be converted into embeddings (Upstage Model).
                    # Chroma is one type of database.

                    retriever = vectorstore.as_retriever(k=2)
                    # Use as_retriever" to load the vectorstore.
                    # (k=2) means that when fetching values for a user's search query, it will retrieve the top 2 most relevant results.
                    # Once the retriever is activated, preprocessing is complete.

                    # Now, an LLM model (Solar_llm) will fetch content when the user searches and generate human-understandable answers.

                    from langchain_upstage import ChatUpstage
                    from langchain_core.messages import HumanMessage, SystemMessage

                    chat = ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"))

                    # Now, the variable chat can hold the model "solar_llm". Setup complete.
                    
                    ## 1) The first step to give the chatbot 'memory'
                    
                    ## Analyze previous messages and the latest user question to rephrase the question in a way that makes sense on its own, without needing context.
                    ## In other words, rephrase the new question so that it focuses only on the question itself.
                    from langchain.chains import create_history_aware_retriever
                    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

                    contextualize_q_system_prompt = """이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다. 
                    이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요. 
                    질문에 답할 필요는 없고, 필요하다면 그저 다시 구성하거나 그대로 두세요."""

                    # MessagePlaceholder: Uses the 'chat_history' input key to include previous message records in the prompt.
                    # The prompt consists of the prompt itself, message history (context information), and the user's question.
                    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", contextualize_q_system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ]
                    )

			        # This is the prompt that rephrases the user's question to improve search results.
                    # For example, if a user has been discussing the benefits of apple juice and then asks, "How much apple juice should I drink in a day?"
                    # Passing this question directly to the database might result in irrelevant responses from the chatbot, so the question needs to be rephrased considering the previous conversation.
                    # The question could be rephrased to something like, "Given the benefits of apple juice are abcd, how much should I drink in a day?"
                    # This prompt is stored in the variable 'contextualize' - which helps the chatbot remember the conversation context.
					
                    # Based on this, create a retriever that remembers the message history.
                    history_aware_retriever = create_history_aware_retriever(
                        # Function to create a chain. (Provided by Langchain) Since there's a template, just need to piece together the code blocks.
                        chat, retriever, contextualize_q_prompt
                    )
                    # From "from langchain.chains" to here is the 1st chain.
                    # Now, based on this chain, a 2nd chain to determine how to answer the actual user's question is needed.

                    ## 2) The second step is to create a retriever chain that can fetch documents using the newly created chain.		
                    from langchain.chains import create_retrieval_chain
                    from langchain.chains.combine_documents import create_stuff_documents_chain

                    qa_system_prompt = """질문-답변 업무를 돕는 보조원입니다. 
                    질문에 답하기 위해 검색된 내용을 사용하세요. 
                    답을 모르면 모른다고 말하세요. 
                    답변은 세 문장 이내로 간결하게 유지하세요.
                    {context}"""
                    qa_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", qa_system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ]
                    )

                    question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
                    # Up to here is the second chain.
                    # The best thing about "Langchain" is that you can create multiple chains and link them together.
                    ## The output includes input, chat_history, context, and answer.
                    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                    # A function that creates a retrieval function for searching. Linked them together. 
                    # Finally, the chain that performs RAG (Retrieval-Augmented Generation)
                    # This will be called every time a question is asked to the chatbot.

                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     

## Name of the Website
st.title("Solar LLM Chatbot")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

## Setup to record the conversation
## In Streamlit, the content will be lost if not activated.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
## To prevent excessive prompt costs
MAX_MESSAGES_BEFORE_DELETION = 4

## Receive user input on the website
# and execute the AI agent created above to get a response.
if prompt := st.chat_input("Ask a question!"):
    
## If it's a question from the user, show the user icon and the question.
     # If there are more than 4 stored conversation records, truncate them.
    if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
        ## Remove the first two messages
        del st.session_state.messages[0]
        del st.session_state.messages[0]  
   
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# If it's a response from the AI, show the AI icon, execute the LLM, get the response, and stream it.
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        result = rag_chain.invoke({"input": prompt, "chat_history": st.session_state.messages})
        # Called rag_chain here and put the user's question into the prompt "input".
        # Also, put the previous conversation records into the "chat_history" to proceed while remembering the conversation context.
        
        ## Show the proof.
        with st.expander("Evidence context"):
            st.write(result["context"])

        for chunk in result["answer"].split(" "):
            full_response += chunk + " "
            time.sleep(0.2)
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})

print("_______________________")
print(st.session_state.messages)