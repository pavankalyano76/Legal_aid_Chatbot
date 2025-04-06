import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, get_response_synthesizer, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer

# Initialize embedding and LLM models
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = Ollama(model="llama3.2:latest", request_timeout=120.0)

# Load the stored index
persist_dir = "C:\\Users\\kavet\\Downloads\\GNU\\persisted_index"
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
index = load_index_from_storage(storage_context)

# Setup retriever and synthesizer
retriever = index.as_retriever()
synthesizer = get_response_synthesizer(response_mode="compact")

# Define custom prompt template
template = (
    "Given the context information and not prior knowledge, "
    "You are a Legal Aid Assistant designed to support users with questions about disability rights and protections. "
    "Provide clear, concise legal guidance based on disability laws, benefits, and related government services. "
    "Reference relevant acts or policy sections when possible, and explain in a way that is easy to understand. "
    "If applicable, give a simple real-life example to help clarify the legal point.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
qa_template = PromptTemplate(template)

class RAGQueryEngine(CustomQueryEngine):
    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        context_str = "\n\n".join([node.get_content() for node in nodes])
        formatted_prompt = qa_template.format(context_str=context_str, query_str=query_str)
        response_obj = self.response_synthesizer.synthesize(query=formatted_prompt, nodes=nodes)
        source_files = list(set([node.metadata.get("file_name", "Unknown Source") for node in nodes]))
        return response_obj, source_files

query_engine = RAGQueryEngine(retriever=retriever, response_synthesizer=synthesizer)

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"You said: {text}")
        return text
    except Exception as e:
        st.error(f"Error: {e}")

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    st.audio(audio_data, format='audio/mp3')

st.title("Legal rights advisor for disabled people")
st.write("Ask questions about disability rights and protections in education, Healthcare, and Employment Dmain. You can type or use speech input.")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

col1, col2 = st.columns(2)

with col1:
    if st.button("🎤 Speak"):
        speech_input = recognize_speech()
        if speech_input:
            st.session_state.messages.append({"role": "user", "content": speech_input})

with col2:
    stop = st.button("⏹️ Stop")

if prompt := st.chat_input("Ask your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            with st.spinner("Thinking..."):
                response, sources = query_engine.custom_query(prompt)
                for chunk in str(response).split():
                    if stop:
                        st.warning("Response stopped by user.")
                        break
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                else:
                    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
                    if not any(greet in prompt.lower() for greet in greetings):
                        full_response += "\n\n**Sources:** " + ", ".join(sources)
                    message_placeholder.markdown(full_response)
                    text_to_speech(str(response))
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Ready for the next question.")

    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.markdown("""
<div style='width: 100%; overflow: hidden;'>
    <marquee behavior="scroll" direction="left" style="color: #555; font-size: small;">
        Note: This response is for general informational purposes only. For accurate and personalized guidance, please contact the ADA National Network at 1-800-949-4232 (V/TTY) or visit www.adata.org.
    </marquee>
</div>
""", unsafe_allow_html=True)
