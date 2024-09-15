import os
import sys
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

from kafka import KafkaProducer, KafkaConsumer
import json

bootstrap_server = ['localhost:9092']
topic = 'message'
producer = KafkaProducer(
    bootstrap_servers=bootstrap_server,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
) # Producer = Local PC


def produce_message(role, content):
    print(f"producing message: {content}", file=sys.stderr)
    message = {
        "role": role,
        "content": content
    }
    producer.send(topic, message)
    producer.flush()


def run_consumer():
    consumer = KafkaConsumer(
        topic,
        group_id='test',
        bootstrap_servers=bootstrap_server,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        # auto_offset_reset='earliest'
    ) # producer 하면 Kafka에 메시지가 들어가는데 우리도 확인 필요 / 확인하려면 Consumer 필요

    print("kafka test consumer is running")
    for message in consumer:
        print(f"Received kafka message: {message.value}")
        time.sleep(0.1)

import threading
consumer_thread = threading.Thread(target=run_consumer)
# consumer_thread.daemon = True 
consumer_thread.start()
# 대화 내용을 카프카 서버(브로커)로 보내서 받은 답변을 확인까지?

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {} # Save the cache of the uploaded PDF file here so that it can remember.

session_id = st.session_state.id
client = None


def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"> </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)


with st.sidebar:
    st.header(f"Add your documents!")

    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        print(uploaded_file)
        try:
            # Create a file_key to save the file_cache.
            file_key = f"{session_id}-{uploaded_file.name}"

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                print("file path:", file_path)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue()) # save the "uploaded file" to a temporary file, open it, and enable indexing.
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):

                    if os.path.exists(temp_dir):
                            print("temp_dir:", temp_dir)
                            loader = PyPDFLoader(file_path)
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    
                    pages = loader.load_and_split()

                    vectorstore = Chroma.from_documents(pages, UpstageEmbeddings(model="solar-embedding-1-large"))

                    retriever = vectorstore.as_retriever(k=2)

                    from langchain_upstage import ChatUpstage
                    from langchain_core.messages import HumanMessage, SystemMessage

                    chat = ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"))

                    from langchain.chains import create_history_aware_retriever
                    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

                    contextualize_q_system_prompt = """이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다. 
                    이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요. 
                    질문에 답할 필요는 없고, 필요하다면 그저 다시 구성하거나 그대로 두세요."""

                    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", contextualize_q_system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ]
                    )

                    history_aware_retriever = create_history_aware_retriever(
                        chat, retriever, contextualize_q_prompt
                    )
                    
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
                    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     

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
        
MAX_MESSAGES_BEFORE_DELETION = 4

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

    produce_message("user", prompt)

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

    produce_message("assistant", full_response)

print("_______________________")
print(st.session_state.messages)