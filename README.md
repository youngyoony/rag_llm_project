# The Most Detailed Project Summary:
http://youngyoon.me/archives/2843

# [GG] Week 1

- Document AI - PDF 문서 등을 GPT에 업로드하면 테이블, 이미지, 차트를 잘 인지하지 못하는 경우가 있음. 그때 도큐먼트를 제대로 인식해서 OCR 등 불러오는 걸 제대로 해 주는 모델
- Solar LLM - 동양권 언어 모델
	- Solar LLM을 활용해서 챗봇
	- Chat: 채팅 활용
	- Embedding: 문서 업로드 했을 때 자연어 언어 모델이 잘 인지할 수 있도록 Embedding 잘해주는 것. 사용자가 질문하는 걸 더 쉽게 찾을 수 있도록
	- Translation: 통역
	- Groundedness Check: 얘가 실제로 어떤 출처를 통해 결과를 냈는지 2차 검증 용도
	- Text-to-SQL: 자연어를 SQL로 변경
	- 아니면 업계별 특화 (Health 등등..)
- Chat API와 Embedding API를 통해서 Rag를 할 수 있는 챗봇을 만들 예정
- Chat API 실행 완료.
- Embedding은 인공지능은 컴퓨터이기 때문에 텍스트 자체로 이해하는 게 아님. Embedding 값이라는 건 인공지능이 잘 이해할 수 있도록 각 단어를 토큰이라는 작은 단위로 → Embedding 숫자로 바꿔준다.
	- Solar embedding large passage: passage는 컨텍스트 리밋이 4000개 정도. PDF 파일을 업로드했을 때 바꿔주는 것. 장문의 PDF, TXT 파일 가능
	- Solar embedding large query: 사용자들이 질문했을 때, 이것도 같이 임베딩.
		- 우리는 이 솔루션을 활용을 해서 챗봇을 만들어 줄 것임
			- 이 Embedding 값들을 벡터스토어에 업로드 - 사용자가 질문했을 때, 그 Embedding 값들과 유사성이 높은 값들을 불러와서 대답하도록 할 것임.
- 솔라 LLM Chatbot을 웹사이트로 만들기 (Rag)
	- Rag: 우리가 문서를 업로드하면 사용자가 쿼리를 보냈을 때 그 질문에 대한 답을 찾고, 그 답변을 LLM 모델을 통해 사용해내는 것
	- 1) Indexing 단계: Indexing은 전처리 단계, 학습시킬만한 문맥 정보를 업로드함.
		- Loading: 로딩을 함. 로딩을 할 수 있는 함수도 되게 많음. PDF 파일에 표나 테이블이 있는지, 한국어로 되어 있는지 등등 특정 문서를 업로드해주는 함수들이 다 다르다. 내 PDF 파일에 맞는 효율적이고 빠른 함수.
		- Split: 올린 파일을 다시 작게 쪼갬. 큰 데이터 파일은 검색하기 어렵고(비효율적, 전부 다 읽어야 함), 모델의 제한된 인풋값 내에 들어가지 못할 가능성이 있음.
		- Store: 쪼갠 데이터를 데이터베이스에 업로드. 컴퓨터가 이해할 수 있는 숫자로 바꿔줌(Embedding) 이것도 엄청 많은 함수가 있는데 Upstage API를 쓸 것임. (Open API, HuggingFace API 등등) 어떤 함수 사용하는지에 따라, 생성AI가 검색을 할 때 그 의미를 더 잘 이해하고 검색할 수 있음.
			- 데이터베이스에 업로드를 하는데, 데이터베이스에 업로드를 할 때에도 DB 종류 많음, 자기가 가지고 있는 검색 알고리즘도 다 다름. 거기다가 Embedding값을 넣는 거고 거기서부터 얼마나 더 잘 찾아주냐는 DB 알고리즘에 달려 있음
	- 2) Retrieval and Generation 단계:
		- Retrieve: 사용자의 검색 쿼리를 통해 DB를 통해 검색
		- Generate: 찾은 답변을 다시 문장으로 LLM이 풀어써서 생성함
- PDF 파일을 하나 업로드 하면 → 전처리 → PDF Preview → 질문하면 답변할 수 있는 챗본 화면

1. 업로드한 PDF 파일을 전처리를 하도록 할 예정. 그런데 얘를 기억하도록 함. 똑같은 PDF 파일을 업로드할 때마다 전처리 과정을 거치면 비용이 급등할 수 있음. (특히 OPEN AI Embedding 같은 경우 전처리 할 때마다 과금) 그래서 기억할 수 있는 Cache 형태를 만들 것임
	```python
	if "id" not in st.session_state:
		st.session_state.id = uuid.uuid4()
		st.session_state.file_cache = {}
	    
	session_id = st.session_state.id
	client = None
	```
2. 모든 것을 다 리셋시키기
	```python
	def reset_chat():
	st.session_state.messages = []
	st.session_state.context = None
	```
3. Display PDF는 프리뷰 형태로 보여주기. 이건 챗봇을 만들 때 필요한 코드는 아닌데 있으면 편하니까 붙인 것.
	```python
	def display_pdf(file):
		#Opening file from file path
	    	
		st.markdown("### PDF Preview")
		base64_pdf = base64.b64encode(file.read()).decode("utf-8)
	    	
		#Embedding PDF in HTML
		pdf_display = f"""<iframe scr="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
												style="height:100vh; width=100%"
											>
											</iframe>"""
		#Displaying File
		st.markdown(pdf_display, unsafe_allow_html=True)
	    	
	```
4. sidebar에 대한 코드가 들어감. 모든 전처리 과정이 여기서 진행됨. 전처리가 다 완료되면 Ready to Chat이라는 화면이 뜰 것임

```python
with st.sidebar: # st는 streamlit이라는 UI/UX 웹사이트 구성. 여기서 사이드바 구성

	st.header(f"add your documents!") # 파일을 업로드할 수 있는 팝업 UX가 자동으로 뜨게 됨
	
	uploaded_file = st.file_uploader("Choose your '.pdf' file", type="pdf") 
	# 파일을 하나 업로드 하면 uploaded_file 안 변수에 저장이 될 것
	
	if uploaded_file:
		print(uploaded_file)
		try:
			file_key = f"{seesion_id}-{uploaded_file.name}" 
			# 얘를 파일 캐시에다가 저장해주기 위해서 (file_cache) 파일 키를 하나 만들어주는 과정임
			
			with tempfile.TemporaryDirectory() as temp_dir:
				file_path = os.path.join(temp_dir, uploaded_file.name)
				print("file path:", file_path)
				
				with open(file_path, "wb") as f:
					f.write(uploaded_file.getvalue()) 
					# 업로드한 파일을 temporary 임시 파일에 저장해서 열어서 indexing 할 수 있게끔 하는 함수
					
				file_key = f"{session_id}-{uploaded_file.name}"
				st.write("Indexing your document...") 
				# 인덱싱 중이라는 걸 사용자에게 알려주는 중
				
				if file_key not in st.session_state.get('file_cache', {}): #마찬가지로 file_cache에 업로드를 해 줌
				
					if os.path.exists(temp_dir): 
						print("temp_dir:", temp_dir)
						loader = PyPDFLoader( 
						# 전처리에서 로딩 해주는 단계. temp_dir라는 임시 파일에 저장되어 있는 파일을 PyPdFLoader에서 그 파일 내용들을 로딩한다.
						# PyPDFLoader는 Langchain의 예제 중에 가장 많이 나오고 있는 예제. 얘가 한국어에 대한 처리나 속도 등등 우수함. 이 Loader 함수를 사용함
						# 문서가 만약에 10장이 있을 경우, 이 Page 안에다가 각 장별로 쪼갠 내용들이 Chunk별로 들어가게 됨 Pages 0, 1, 2, ... 장 수만큼
							file_path
						)
					else:
						st.error('Could not find the file you uploaded, please check again...')
					
					pages = loader.load_and_split() 
					# 업로드한 함수를 쪼개 줄 것임. 그 쪼개는 함수를 load_and_split이라는 함수 사용하게 됨. 쪼갠 내용을 pages에 저장
					# 1장만 있는 걸 쓸 거기 때문에 구체적으로 더 잘라 줄 필요는 없을 것 같다.
					
					vectorstore = Chroma.from_documents(pages, UpstageEmbeddings(model="solar-embedding-1-large")) 
					# 쪼갠 내용을 pages에 저장, 그걸 Embedding값으로 변환해야 하는데 그걸 Upstage Model로 변환
					# Chroma는 Database 중에서 한 가지임. 이걸 vectorstore에 저장
					
					retriever = vectorstore.as_retriever(k=2) 
					# vectorstore을 불러올 수 있는 retriever이라는 함수를 사용, 변수 저장.
					# k=2 사용자 검색 쿼리 vs 값을 불러올 때 유사성이 높은 애들 (1~n 순위) 그 중에서 top 2만 들고 오겠다.
					# retriever 활성화 하면 전처리 끝
					
					# 이제 실제로 사용자가 검색했을 때 내용을 불러오고, 사람이 이해할 수 있는 답변으로 생성해 주는 LLM 모델
					# Solar_llm
					
					from langchain_upstage import ChatUpstage
					from langchain_core.messages import HumanMessage, SystemMessage
					
					chat = ChatUpstage_api_key = os.getenv("up_P2q3pHb9pW6PVmp0m5dY0S58yplVQ"))
					
					# 이제 chat이라는 변수 안에 solar_llm 이라는 모델을 넣을 수 있음. 셋업 완료
					
					## 1) 챗봇에 '기억'을 입히기 위한 첫 번째 단계
					
					## 이전의 메시지들과 최신 사용자 질문을 분석해, 문맥에 대한 정보가 없어 혼자서만 봤을 때 이해할 수 있도록 질문을 다시 구성함
					## 즉, 새로 들어온 그 질문 자체에만 집중할 수 있도록 다시 재편성
					
					from langchain.chains import create_history_aware_retriever
					from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
					
					contextualize_q_system_prompt = """이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다.
					이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요.
					질문에 답할 필요는 없고, 필요하다면 그저 다시 구성하거나 그대로 두세요."""
					# MessagePlaceholder: 'chat_history' 입력 키를 사용하여 이전 메시지 기록들을 프롬프트에 포함시킴.
					# 즉 프롬프트, 메시지 기록 (문맥 정보), 사용자의 질문으로 프롬프트가 구성됨.
					contextualize_q_prompt = ChatPromptTemplate.from_messages(
						[
							("system", contextualize_q_system_prompt),
							MessagesPlaceholder("chat_history"),
							("human", "{input}"),
						]
					)
					# 여기까지 검색을 좀 더 잘할 수 있도록 사용자의 질문을 재구성해주는 프롬프트
					# 예: 내가 사과주스 효능에 관련된 대화를 했다가 - 사용자가 사과주스 하루에 얼마나 먹어야 해? 
					# - 이 질문을 그대로 데이터베이스에 넘기면 딴소리 하기 때문에, 예전 대화 내용을 기억하면서 문장 재구성해야 하니까.
					# - 그래서 사과주스 효능은 abcd니까 하루에 얼마나 먹어야 해? 이런 식으로 질문을 재구성해 줌
					# 이 프롬프트가 contextualize 라는 변수에 저장이 되어 있는 것이 - 이 토대로 대화 내용을 기억할 수 있음
	
					
					# 이를 토대로 메시지 기록을 기억하는 retriever를 생성
					history_aware_retreiver = create_history_aware_retriever( 
					# Chain을 만들어 주는 함수. 이게 Langchain에서 만들어주는 함수. 템플릿만, 코드블럭을 이어붙이면 됨
						chat, retriever, contextualize_q_prompt
					)
					# from langchain.chains부터 여기까지 1번째 Chain임
					# 이제 그러면 이 Chain을 토대로, 실제 사용자의 질문에 대한 답을 받아왔을 때 어떤 식으로 답변할 건지 알려주는 2번째 Chain 필요
					
					# 2) 두 번째 단계로, 방금 전 생성한 체인을 사용하여 문서를 불러올 수 있는 retriever 체인을 생성한다.
					from langchain.chains import create_retrieval_chain
					from langchain.chains.combine_documents import create_stuff_documents_chain
					
					qa_system_prompt = """질문-답변 업무를 돕는 보조원입니다. 
					질문에 답하기 위해 검색된 내용을 사용하세요. 
					답을 모르면 모른다고 말하세요. 
					답변은 세 문장 이내로 간결하게 유지하세요.
					
					{context}"""
					qa_prompt = ChatPromptTemplate.from_message(
						[
							("system", qa_system_prompt),
							MessagesPlaceholder("chat_history"),
							("human", "{input}"),
						]
					)
					
					question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
					# 여기까지 2번째 Chain
					# Langchain이 좋은 이유는, 여러가지 Chain을 만들면 이걸 이어붙일 수 있기 때문임.
					## 결과값은 input, chat_history, context, answer 포함함.
					rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
					# 검색을 할 수 있는 retrieval 함수를 만들어주는 함수! 이렇게 이어붙임. 마지막으로 rag를 할 수 있는 chain!
					# chatbot한테 질문을 할 때마다 한 번씩 호출하게 만듦
				
				st.success("Ready to Chat!")
				display_pdf(uploaded_file)
			except Exception as e:
				st.error(f"An error occurred: {e}")
				st.stop()
							
	
	# 웹사이트 제목 
	st.title("Solar LLM Chatbot")
	
	if "openai_model" not in st.session_state:
		st.session_state["openai_model"] = "gpt-3.5-turbo"
		
	if "messages" not in st.session_state:
		st.session_state.messages = []
		
	## 대화 내용을 기록하기 위해 셋업
	## Streamlit 특성상 활성화하지 않으면 내용이 다 날아감.
	for message in st.session_state.messages:
		with st.chat_message(message(["role"]):
			st.markdown(message["content"])
			
	## 프롬프트 비용이 너무 많이 소요되는 것을 방지하기 위해
	MAX_MESSAGES_BEFORE_DELETION = 4
	
	## 웹사이트에서 유저의 인풋을 받고 위에서 만든 AI 에이전트 실행시켜 답변 받기
	if prompt := st.chat_input("Ask a question!"):
	
	## 유저가 보낸 질문이면 유저 아이콘과 질문 보여주기
	## 만일 현재 저장된 대화 내용 기록이 4개보다 많으면 자르기
		if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
			## Remove the first two messages
			del st.session_state.messages[0]
			del st.session_state.messages[0]
			
		st.session_state.messages.append({"role": "user", "content": prompt})
		with st.chat_message("user"):
			st.markdown(prompt)
			
	## AI가 보낸 답변이면 AI 아이콘이랑 LLM 실행시켜서 답변 받고 스트리밍해서 보여주기
		with st.chat_message("assistant"):
			message_placeholder = st.empty()
			full_response = ""
			
			result = rag_chain.invoke({"input": prompt, "chat_history": st.session_state.messages})
			# 실제로 여기서 rag_chain을 호출해서 사용자의 질문을 prompt input으로 넣어두고, 
			# 전 대화 기록들도 chat_history 변수에 넣어서 대화 내용을 기억한 상태로 진행 중
			
			## 증거자료 보여주기
			with st.expander("Evidence context"):
				st.write(resulte["context"])
				
			for chunk in result["answer"].split(" "):
				full_response += chunk + " "
				time.sleep(0.2)
				message_placeholder.markdown(full_response + " ")
				message_placeholder.markdown(full_response)
				
			st.session_state.messages.append({"role": "assistant", "content"): full_response})
			
		print("_________________")
		print(st.session_state.messages)
		

```
