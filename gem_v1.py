import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import requests
import wavio
import numpy as np
import base64
import time
import webrtcvad
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, Document
import traceback
import asyncio
import websockets
import json
from scipy import signal
import io
import wave
from enum import Enum

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
GPU_INSTANCE_IP = "13.126.45.55"
GPU_INSTANCE_PORT = "8080"
LLM_API_BASE_URL = f"http://{GPU_INSTANCE_IP}:{GPU_INSTANCE_PORT}/v1"
SARVAM_API_KEY = "sk_q9v0zex9_5pnlS75CWasgXVmUhmDbwvS6"
CURRENT_MODEL_ID_FROM_SERVER = "dolphin-2.9-llama3-8b-q8_0.gguf"
COMPANY_NAME_FOR_LLM = "Zype"
AI_NAME = "Anushka"
LLM_MAX_TOKENS = 150

# Simple audio settings
AUDIO_SAMPLE_RATE = 16000  # Sarvam's sample rate
EXOTEL_SAMPLE_RATE = 8000  # Exotel's sample rate
CHUNK_DURATION_MS = 20  # 20ms chunks
CHUNK_SIZE = int(EXOTEL_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# Simple recording settings
FIXED_RECORDING_DURATION = 4.0  # Record for 4 seconds max
SILENCE_THRESHOLD = 0.005  # RMS threshold for silence
SILENCE_DURATION_TO_STOP = 1.5  # Stop recording after 1.5s of silence
MIN_RECORDING_DURATION = 0.5  # Minimum recording time

WEBSOCKET_PORT = 8765

# Language settings
DEFAULT_USER_LANGUAGE = "hi-IN"
TTS_SPEAKER_NAME = "anushka"

# RAG settings
RAG_SIMILARITY_THRESHOLD = 0.85


# Simple call states
class CallState(Enum):
    WAITING = "waiting"
    AI_SPEAKING = "ai_speaking"
    LISTENING = "listening"
    PROCESSING = "processing"
    ENDED = "ended"


# --- Knowledge Base for RAG (IN ENGLISH) ---
knowledge_base_entries_english = [
    {
        "text": "Uploading your bank statement helps process your loan application faster and can lead to a better credit line. It helps us understand your financial situation.",
        "source": "benefit_of_upload"},
    {
        "text": "You need to upload the bank statement for your salary account, the account where your salary is credited.",
        "source": "which_account"},
    {"text": "You need to upload the bank statement for the last 4 months.", "source": "period_of_statement"},
    {
        "text": "There are several ways to upload your bank statement in the Zype app: through an Account Aggregator, using Netbanking, or by directly uploading a PDF file.",
        "source": "how_to_upload"},
    {
        "text": "All these upload options are available within the Zype app. You can upload it at the designated spot in the app.",
        "source": "where_to_upload"},
]
DEFAULT_FALLBACK_PASSAGE_ENGLISH = "You can easily upload your statement in the Zype app."

# --- RAG Setup (Operates on English) ---
embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
print(f"Using embedding model: {embedding_model_name}")
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cpu'})

documents_for_vectorstore = [
    Document(page_content=entry["text"], metadata={"source": entry["source"]})
    for entry in knowledge_base_entries_english
]
vector_store = None
try:
    print("Initializing RAG vector store (English)...")
    vector_store = FAISS.from_documents(documents_for_vectorstore, embeddings)
    print("RAG vector store initialized successfully.")
except Exception as e:
    print(f"FATAL: Could not initialize RAG vector store: {e}")
    traceback.print_exc()


# --- Translation and STT Functions ---
def speech_to_english_text(audio_data):
    """Convert audio to English text using Sarvam STT-Translate"""
    temp_file = "temp_audio_stt.wav"
    try:
        wavio.write(temp_file, audio_data, AUDIO_SAMPLE_RATE, sampwidth=2)

        url = "https://api.sarvam.ai/speech-to-text-translate"
        headers = {"api-subscription-key": SARVAM_API_KEY}
        files = {"file": (temp_file, open(temp_file, "rb"), "audio/wav")}
        data = {"model": "saaras:v2.5"}

        response = requests.post(url, headers=headers, files=files, data=data, timeout=15)
        response.raise_for_status()

        result_json = response.json()
        english_transcript = result_json.get("transcript", "")
        detected_lang = result_json.get("language_code", DEFAULT_USER_LANGUAGE)

        print(f"STT-Translate: Success. EN Text: '{english_transcript}', Detected Lang: {detected_lang}")

        # Basic validation
        if not english_transcript.strip():
            return "", DEFAULT_USER_LANGUAGE

        words = english_transcript.split()
        if len(words) > 25:
            print(f"Transcript too long ({len(words)} words), rejecting")
            return "", detected_lang

        return english_transcript.strip(), detected_lang

    except Exception as e:
        print(f"STT-Translate Error: {e}")
        return "", DEFAULT_USER_LANGUAGE
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def translate_text_sarvam(text_to_translate: str, target_language_code: str = DEFAULT_USER_LANGUAGE):
    """Translate English text to target language"""
    if not text_to_translate.strip() or target_language_code.startswith("en"):
        return text_to_translate

    url = "https://api.sarvam.ai/translate"
    headers = {"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"}
    data = {
        "input": text_to_translate,
        "source_language_code": "en-IN",
        "target_language_code": target_language_code,
        "model": "mayura:v1",
        "speaker_gender": "Female",
        "mode": "modern-colloquial"
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()

        translated_text = response.json().get("translated_text", "")
        if translated_text:
            print(f"Translate: '{text_to_translate[:30]}...' -> '{translated_text[:30]}...' ({target_language_code})")
            return translated_text
        else:
            print(f"Translate Error: Empty translated_text field")
            return text_to_translate

    except Exception as e:
        print(f"Translate Error: {e}")
        return text_to_translate


def get_audio_from_text_with_language(english_text: str, target_language_code: str):
    """Generate audio from English text by translating and using TTS"""
    if not english_text or not english_text.strip():
        return np.zeros(int(AUDIO_SAMPLE_RATE * 0.1), dtype=np.int16), AUDIO_SAMPLE_RATE

    # Translate to target language first
    translated_text = translate_text_sarvam(english_text, target_language_code)

    # Generate TTS
    url = "https://api.sarvam.ai/text-to-speech"
    headers = {"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"}
    data = {
        "text": translated_text,
        "target_language_code": target_language_code,
        "speaker": TTS_SPEAKER_NAME,
        "speech_sample_rate": AUDIO_SAMPLE_RATE,
        "audio_format": "wav",
        "enable_preprocessing": True
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        response.raise_for_status()
        response_data = response.json()

        if "audios" not in response_data or not response_data["audios"]:
            raise ValueError("No audio data in TTS response")

        audio_bytes = base64.b64decode(response_data["audios"][0])
        with io.BytesIO(audio_bytes) as f:
            with wave.open(f, 'rb') as wf:
                sample_rate = wf.getframerate()
                audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

        print(f"TTS successful ({target_language_code}): '{translated_text[:30]}...'")
        return audio_data, sample_rate

    except Exception as e:
        print(f"TTS Error: {e}")
        # Fallback to English TTS if target language fails
        if target_language_code != "en-IN":
            print("TTS Fallback: Trying English")
            return get_audio_from_text_with_language(english_text, "en-IN")

        # Return silence if everything fails
        return np.zeros(int(AUDIO_SAMPLE_RATE * 0.1), dtype=np.int16), AUDIO_SAMPLE_RATE


# --- RAG Retrieval Function ---
def retrieve_passages_rag(query_english: str, vec_store_instance, default_passage_english: str,
                          threshold=RAG_SIMILARITY_THRESHOLD):
    """Retrieve relevant passages using RAG"""
    if not vec_store_instance or not query_english.strip():
        return default_passage_english

    try:
        results_with_scores = vec_store_instance.similarity_search_with_score(query_english, k=1)
        if results_with_scores:
            doc, score = results_with_scores[0]
            print(
                f"RAG: Retrieved: '{doc.page_content}' (Source: {doc.metadata.get('source', 'N/A')}, Score: {score:.4f})")

            if score > threshold:
                print(f"RAG: Score {score:.4f} > threshold {threshold}. Using fallback.")
                return default_passage_english

            return doc.page_content

        return default_passage_english

    except Exception as e:
        print(f"RAG Error: {e}")
        traceback.print_exc()
        return default_passage_english


def post_process_llm_english_response(english_response: str, user_english_input: str):
    """Post-process LLM English response"""
    response = english_response.strip()
    response = response.replace('–', '')
    response = re.sub(r"'", '', response)

    if not response:
        return DEFAULT_FALLBACK_PASSAGE_ENGLISH

    # Limit to 2 sentences
    sentences = [s.strip() for s in response.split('.') if s.strip()]
    if len(sentences) > 2:
        final_response = ". ".join(sentences[:2]) + "."
    elif sentences:
        final_response = ". ".join(sentences)
        if not final_response.endswith('.'):
            final_response += "."
    else:
        final_response = response
        if not final_response.endswith('.'):
            final_response += "."

    # Check for prohibited phrases
    prohibited_phrases_english = ["any other questions", "further assistance", "feel free to ask"]
    if any(phrase.lower() in final_response.lower() for phrase in prohibited_phrases_english):
        print(f"Prohibited phrase in LLM EN response. Replacing with fallback.")
        final_response = DEFAULT_FALLBACK_PASSAGE_ENGLISH

    return final_response.strip()


# --- Audio Generation Functions ---
def generate_ai_greeting_and_identity_check(customer_name: str) -> str:
    return f"Hello, am I speaking with {customer_name}?"


def generate_ai_self_introduction() -> str:
    return f"This is {AI_NAME} calling from {COMPANY_NAME_FOR_LLM}."


def generate_ai_main_purpose_statement():
    return "We've noticed that the salary bank statement for your loan application hasn't been uploaded yet. Please upload it as soon as possible."


system_instruction_english = f"""
You are {AI_NAME}, a polite and professional assistant from {COMPANY_NAME_FOR_LLM}. Your primary goal is to remind the customer to upload their pending salary bank statement for their loan application and try to get an ETA. Keep your responses concise (1-2 short sentences).

**Instructions**:
- You will receive the user's input (translated to English) and relevant context from a knowledge base (English).
- Strictly use the provided context. If context is "{DEFAULT_FALLBACK_PASSAGE_ENGLISH}" or irrelevant, use a general statement like "You can upload your statement in the {COMPANY_NAME_FOR_LLM} app."
- Responses MUST be in polite, conversational English, ending with a period (.). Aim for 1-2 short sentences.
- After addressing the query, if the user seems ready (e.g., "Okay", "Alright"), ask for an ETA if not provided. Example: "Okay. So, will you be able to do it by this evening?"
- If user provides ETA (e.g., "I'll do it today"), acknowledge and close. Example: "That's great. We'll wait for your statement by [ETA]. Thank you! Have a good day."
- If input is unclear: "I'm sorry, I didn't quite catch that. Could you please repeat?" After two unclear, end: "It seems there might be an issue with the line. We'll try to contact you later. Thank you."
- If user asks why: Use context, or explain: "Uploading the bank statement helps in processing your loan application faster."

**Call Flow**:
1. System: Identity check
2. System: Self introduction  
3. System: Purpose statement
4. You handle user responses based on context

**Strict Rules**: 
- NO "Is there anything else...", NO PII requests, NO financial advice, NO arguing
- If abusive, disengage: "I'm unable to continue this conversation. Thank you, have a good day."
- Preferred closing: "Thank you! Have a good day."
"""

# --- LLM Setup ---
llm = ChatOpenAI(
    openai_api_base=LLM_API_BASE_URL,
    openai_api_key="NotNeeded",
    model_name=CURRENT_MODEL_ID_FROM_SERVER,
    temperature=0.5,
    max_tokens=LLM_MAX_TOKENS,
)


# --- Enhanced Call Manager with Translation ---
class EnhancedCallManager:
    def __init__(self, customer_name="हीरा"):
        self.customer_name = customer_name
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="history")
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_instruction_english),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        self.conversation = ConversationChain(
            llm=llm,
            memory=self.memory,
            prompt=self.prompt_template,
            verbose=False
        )

        # Simple state
        self.state = CallState.WAITING
        self.conversation_stage = "greeting"  # greeting, introduction, conversation
        self.unclear_input_count = 0

        # Language management
        self.current_user_language = DEFAULT_USER_LANGUAGE
        self.language_locked = False

        # Simple recording state
        self.is_recording = False
        self.audio_buffer = []
        self.recording_start_time = None
        self.last_audio_time = None

    def start_recording(self):
        print("Started recording")
        self.is_recording = True
        self.audio_buffer = []
        self.recording_start_time = time.time()
        self.last_audio_time = time.time()

    def should_stop_recording(self):
        if not self.is_recording:
            return False

        current_time = time.time()
        recording_duration = current_time - self.recording_start_time
        silence_duration = current_time - self.last_audio_time

        # Stop if max duration reached
        if recording_duration >= FIXED_RECORDING_DURATION:
            print(f"Max duration reached ({recording_duration:.1f}s)")
            return True

        # Stop if silent for too long (but only after minimum recording time)
        if (recording_duration >= MIN_RECORDING_DURATION and
                silence_duration >= SILENCE_DURATION_TO_STOP):
            print(f"Silence timeout ({silence_duration:.1f}s)")
            return True

        return False

    def stop_recording(self):
        if self.is_recording:
            duration = time.time() - self.recording_start_time
            print(f"Stopped recording ({duration:.1f}s, {len(self.audio_buffer)} chunks)")
            self.is_recording = False
            return True
        return False


async def process_audio_chunk(audio_chunk, call_manager, websocket):
    """Simple audio processing - just collect chunks and check for silence"""

    # Only process if we're listening
    if call_manager.state != CallState.LISTENING:
        return

    # Calculate audio level
    audio_level = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2)) / 32768.0

    # Start recording on first audio above threshold
    if not call_manager.is_recording:
        if audio_level > SILENCE_THRESHOLD:
            call_manager.start_recording()
        else:
            return  # Skip silent chunks when not recording

    # Add chunk to buffer
    call_manager.audio_buffer.append(audio_chunk)

    # Update last audio time if above threshold (for silence detection)
    if audio_level > SILENCE_THRESHOLD:
        call_manager.last_audio_time = time.time()

    # Check if we should stop recording
    if call_manager.should_stop_recording():
        await process_recorded_audio(call_manager, websocket)


async def process_recorded_audio(call_manager, websocket):
    """Process the recorded audio buffer with translation"""
    call_manager.stop_recording()
    call_manager.state = CallState.PROCESSING

    user_english_input = ""
    detected_lang = DEFAULT_USER_LANGUAGE

    if call_manager.audio_buffer:
        try:
            # Combine all audio chunks
            combined_audio = np.concatenate(call_manager.audio_buffer)

            # Check if we have enough audio
            duration = len(combined_audio) / EXOTEL_SAMPLE_RATE
            if duration < MIN_RECORDING_DURATION:
                print(f"Recording too short ({duration:.1f}s), ignoring")
                call_manager.state = CallState.LISTENING
                return

            # Resample to 16kHz for STT
            audio_16k = signal.resample(
                combined_audio,
                int(len(combined_audio) * AUDIO_SAMPLE_RATE / EXOTEL_SAMPLE_RATE)
            )

            print(f"Processing {duration:.1f}s of audio")
            user_english_input, detected_lang = speech_to_english_text(audio_16k.astype(np.int16))

            # Language lock-in logic
            if not call_manager.language_locked and call_manager.conversation_stage == "conversation":
                call_manager.current_user_language = detected_lang
                call_manager.language_locked = True
                print(f"LANGUAGE LOCKED TO: {call_manager.current_user_language}")

        except Exception as e:
            print(f"Audio processing error: {e}")

    if not user_english_input or not user_english_input.strip():
        user_english_input = "User was silent or input was unclear."
        call_manager.unclear_input_count += 1
        print(f"No transcript (count: {call_manager.unclear_input_count})")
    else:
        print(f"User (EN): '{user_english_input}', Detected Lang: {detected_lang}")
        call_manager.unclear_input_count = 0

    # Add to memory (English version for LLM)
    call_manager.memory.chat_memory.add_user_message(user_english_input)

    # Handle based on conversation stage
    if call_manager.conversation_stage == "greeting":
        await handle_greeting_response(user_english_input, call_manager, websocket)
    elif call_manager.conversation_stage == "conversation":
        await handle_conversation(user_english_input, call_manager, websocket)


async def handle_greeting_response(user_english_input, call_manager, websocket):
    """Handle identity confirmation"""
    pos_words = ["yes", "speaking", "it is", "that's me", call_manager.customer_name.lower()]
    neg_words = ["no", "wrong number", f"not {call_manager.customer_name.lower()}"]

    user_lower = user_english_input.lower()

    if any(word in user_lower for word in neg_words):
        await say_and_end(websocket, "My apologies, it seems I have the wrong number. Thank you.", call_manager)
        return

    # Continue to introduction
    call_manager.conversation_stage = "introduction"
    await say_to_user(websocket, generate_ai_self_introduction(), call_manager)

    # Wait a bit then give purpose
    await asyncio.sleep(0.5)
    purpose = generate_ai_main_purpose_statement()
    await say_to_user(websocket, purpose, call_manager)

    call_manager.conversation_stage = "conversation"


async def handle_conversation(user_english_input, call_manager, websocket):
    """Handle main conversation with RAG and translation"""
    global vector_store

    # Too many unclear inputs
    if call_manager.unclear_input_count >= 3:
        await say_and_end(websocket,
                          "It seems there might be an issue with the line. We'll try to contact you later. Thank you.",
                          call_manager)
        return

    # Get RAG context (operates on English)
    retrieved_passage_en = DEFAULT_FALLBACK_PASSAGE_ENGLISH
    if user_english_input.strip() and "User was silent" not in user_english_input and vector_store:
        retrieved_passage_en = retrieve_passages_rag(
            user_english_input,
            vector_store,
            DEFAULT_FALLBACK_PASSAGE_ENGLISH
        )

    # Prepare LLM input (all in English)
    llm_input = f"User utterance: \"{user_english_input}\"\nRelevant context: \"{retrieved_passage_en}\""
    print(f"LLM Input (EN): '{llm_input[:200]}...'")

    try:
        # Generate LLM response (in English)
        llm_raw_response = call_manager.conversation.predict(input=llm_input)
        llm_response_en = post_process_llm_english_response(llm_raw_response, user_english_input)

        print(f"LLM Raw (EN): {llm_raw_response}")
        print(f"LLM Processed (EN): {llm_response_en}")

        # Check if closing
        closing_words = ["thank you! have a good day", "thanks for your time", "we'll try to contact you later"]
        is_closing = any(word in llm_response_en.lower() for word in closing_words)

        # Check for ETA acknowledgment
        eta_keywords = ["today", "tomorrow", "evening", "morning", "will do", "i'll upload", "i will upload"]
        user_gave_eta = any(kw in user_english_input.lower() for kw in eta_keywords) and \
                        not any(
                            neg in user_english_input.lower() for neg in ["won't", "can't", "not able", "not today"])

        llm_acked_eta = ("we'll wait" in llm_response_en.lower() or "look forward" in llm_response_en.lower()) and \
                        ("thank you" in llm_response_en.lower() or "have a good day" in llm_response_en.lower())

        if user_gave_eta and not llm_acked_eta:
            # Force ETA acknowledgment
            eta_response = "That's great. We'll wait for your statement. Thank you! Have a good day."
            await say_and_end(websocket, eta_response, call_manager)
            return

        if is_closing or llm_acked_eta:
            await say_and_end(websocket, llm_response_en, call_manager)
        else:
            await say_to_user(websocket, llm_response_en, call_manager)

    except Exception as e:
        print(f"LLM Error: {e}")
        await say_and_end(websocket, "I'm facing a technical issue. We'll connect later. Thank you.", call_manager)


async def say_to_user(websocket, english_text, call_manager):
    """Say something in user's language and return to listening"""
    call_manager.state = CallState.AI_SPEAKING
    print(f"AI (EN): '{english_text}'")

    try:
        # Generate audio in user's language
        audio_data, rate = get_audio_from_text_with_language(english_text, call_manager.current_user_language)
        call_manager.memory.chat_memory.add_ai_message(english_text)
        await stream_audio_to_exotel(websocket, audio_data, rate)

        # Brief pause then start listening
        await asyncio.sleep(0.3)
        call_manager.state = CallState.LISTENING
        print("Listening...")

    except Exception as e:
        print(f"AI speech error: {e}")
        call_manager.state = CallState.LISTENING


async def say_and_end(websocket, english_text, call_manager):
    """Say something in user's language and end the call"""
    call_manager.state = CallState.AI_SPEAKING
    print(f"AI (EN, closing): '{english_text}'")

    try:
        # Generate audio in user's language
        audio_data, rate = get_audio_from_text_with_language(english_text, call_manager.current_user_language)
        call_manager.memory.chat_memory.add_ai_message(english_text)
        await stream_audio_to_exotel(websocket, audio_data, rate)

    except Exception as e:
        print(f"AI speech error: {e}")

    call_manager.state = CallState.ENDED


async def stream_audio_to_exotel(websocket, audio_data, sample_rate):
    """Stream audio data to Exotel"""
    # Resample to 8kHz if needed
    if sample_rate != EXOTEL_SAMPLE_RATE:
        audio_data_8k = signal.resample(
            audio_data,
            int(len(audio_data) * EXOTEL_SAMPLE_RATE / sample_rate)
        )
    else:
        audio_data_8k = audio_data

    # Ensure int16
    audio_data_8k = audio_data_8k.astype(np.int16)

    # Send in chunks
    chunk_size = CHUNK_SIZE
    for i in range(0, len(audio_data_8k), chunk_size):
        chunk = audio_data_8k[i:i + chunk_size]

        # Pad if needed
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')

        # Convert and send
        audio_bytes = chunk.tobytes()
        payload = base64.b64encode(audio_bytes).decode('utf-8')

        try:
            await websocket.send(json.dumps({
                'event': 'media',
                'media': {'payload': payload}
            }))
            await asyncio.sleep(0.02)  # Small delay
        except Exception as e:
            print(f"Send error: {e}")
            break


# --- Main WebSocket Handler ---
async def handle_exotel_call(websocket):
    """Handle Exotel WebSocket connection"""
    print(f"New call connected")
    call_manager = EnhancedCallManager()

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                event_type = data.get('event')

                if event_type == 'connect':
                    print("Connected")

                elif event_type == 'start':
                    print("Call started")

                    # Start with greeting
                    await asyncio.sleep(0.3)
                    await say_to_user(
                        websocket,
                        generate_ai_greeting_and_identity_check(call_manager.customer_name),
                        call_manager
                    )

                elif event_type == 'media':
                    if call_manager.state == CallState.LISTENING:
                        payload = data.get('media', {}).get('payload', '')
                        if payload:
                            try:
                                audio_data = base64.b64decode(payload)
                                audio_chunk = np.frombuffer(audio_data, dtype=np.int16)
                                await process_audio_chunk(audio_chunk, call_manager, websocket)
                            except Exception as e:
                                print(f"Media processing error: {e}")

                elif event_type == 'stop':
                    print("Call stopped")
                    break

            except json.JSONDecodeError as e:
                print(f"JSON error: {e}")
            except Exception as e:
                print(f"Message error: {e}")

    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")
    except Exception as e:
        print(f"Call error: {e}")
    finally:
        print("Call ended")


async def main():
    """Start the server"""
    print(f"Starting Enhanced Exotel server on ws://localhost:{WEBSOCKET_PORT}")
    print(f"Configuration:")
    print(f"  - Translation pipeline: STT->EN->LLM->TL->TTS")
    print(f"  - RAG enabled: {vector_store is not None}")
    print(f"  - Default language: {DEFAULT_USER_LANGUAGE}")
    print(f"  - Max recording: {FIXED_RECORDING_DURATION}s")
    print(f"  - Silence timeout: {SILENCE_DURATION_TO_STOP}s")

    async with websockets.serve(handle_exotel_call, "localhost", WEBSOCKET_PORT):
        print("Enhanced server ready with translation and RAG!")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Server error: {e}")
        traceback.print_exc()