import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import requests
import wavio
import numpy as np
import base64
import time
from collections import deque
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
import traceback
import asyncio
import websockets
import json
from scipy import signal
import io
import wave
from enum import Enum

# --- Configuration ---
GPU_INSTANCE_IP = "13.126.45.55"
GPU_INSTANCE_PORT = "8080"
LLM_API_BASE_URL = f"http://{GPU_INSTANCE_IP}:{GPU_INSTANCE_PORT}/v1"
SARVAM_API_KEY = "sk_q9v0zex9_5pnlS75CWasgXVmUhmDbwvS6"
CURRENT_MODEL_ID_FROM_SERVER = "dolphin-2.9-llama3-8b-q8_0.gguf"
COMPANY_NAME_FOR_LLM = "Zype"
AI_NAME = "Anushka"
LLM_MAX_TOKENS = 150

# Enhanced audio settings with VAD
AUDIO_SAMPLE_RATE = 16000  # Sarvam's sample rate
EXOTEL_SAMPLE_RATE = 8000  # Exotel's sample rate
CHUNK_DURATION_MS = 20  # 20ms chunks
CHUNK_SIZE = int(EXOTEL_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# Enhanced VAD settings
FIXED_RECORDING_DURATION = 8.0
VOICE_THRESHOLD = 0.008  # RMS threshold for voice detection (slightly higher)
SILENCE_THRESHOLD = 0.003  # RMS threshold for silence (lower for better detection)
SILENCE_DURATION_TO_STOP = 1.2  # Stop recording after 1.2s of silence
MIN_RECORDING_DURATION = 0.3  # Minimum recording time
PRE_SPEECH_BUFFER_SIZE = 5  # Number of chunks to keep before speech detection
POST_AI_SPEECH_DELAY = 0.1  # Delay before starting to listen after AI speaks
NOISE_FLOOR_ADAPTATION_RATE = 0.1  # Rate at which noise floor adapts

# VAD window settings
VAD_WINDOW_SIZE = 3  # Number of consecutive chunks to confirm voice activity
VAD_CONFIRMATION_THRESHOLD = 2  # Minimum chunks in window that must be above threshold
WEBSOCKET_PORT = 8765

# Language settings
DEFAULT_USER_LANGUAGE = "hi-IN"
TTS_SPEAKER_NAME = "anushka"
customer_name = "हीरा"
ALLOWED_LANGUAGES = ["en-IN", "hi-IN", "kn-IN", "te-IN", "ta-IN"]

class CallState(Enum):
    WAITING = "waiting"
    AI_SPEAKING = "ai_speaking"
    LISTENING = "listening"
    PROCESSING = "processing"
    POST_AI_DELAY = "post_ai_delay"  # New state for post-AI speech delay
    ENDED = "ended"


# VAD Class for better voice activity detection
class VoiceActivityDetector:
    def __init__(self):
        self.noise_floor = SILENCE_THRESHOLD
        self.recent_chunks = deque(maxlen=VAD_WINDOW_SIZE)
        self.pre_speech_buffer = deque(maxlen=PRE_SPEECH_BUFFER_SIZE)

    def reset(self):
        """Reset VAD state"""
        self.recent_chunks.clear()
        self.pre_speech_buffer.clear()

    def clear_pre_speech_buffer(self):
        """Clear the pre-speech buffer"""
        self.pre_speech_buffer.clear()

    def update_noise_floor(self, audio_level):
        """Adaptively update noise floor"""
        if audio_level < self.noise_floor * 2:  # Only update with quiet audio
            self.noise_floor = (self.noise_floor * (1 - NOISE_FLOOR_ADAPTATION_RATE) +
                                audio_level * NOISE_FLOOR_ADAPTATION_RATE)

    def is_voice_activity(self, audio_chunk):
        """Detect voice activity in audio chunk"""
        audio_level = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2)) / 32768.0

        # Update noise floor during quiet periods
        self.update_noise_floor(audio_level)

        # Dynamic threshold based on noise floor
        dynamic_threshold = max(VOICE_THRESHOLD, self.noise_floor * 3)

        # Check if current chunk has voice activity
        is_voice = audio_level > dynamic_threshold

        # Add to recent chunks window
        self.recent_chunks.append(is_voice)

        # Always add to pre-speech buffer for potential voice start
        self.pre_speech_buffer.append(audio_chunk)

        # Confirm voice activity if enough recent chunks are above threshold
        if len(self.recent_chunks) >= VAD_CONFIRMATION_THRESHOLD:
            voice_count = sum(self.recent_chunks)
            return voice_count >= VAD_CONFIRMATION_THRESHOLD

        return False

    def get_pre_speech_audio(self):
        """Get buffered pre-speech audio"""
        if self.pre_speech_buffer:
            return list(self.pre_speech_buffer)
        return []


# --- System Prompt with Knowledge Base Information ---
system_instruction_english = f"""
You are {AI_NAME}, a polite and professional assistant from {COMPANY_NAME_FOR_LLM}.You are talking to {customer_name}. Your primary goal is to remind the customer to upload their pending salary bank statement for their loan application and try to get an ETA. Keep your responses concise (1-2 short sentences).

**Instructions**:
- Use the following information to answer user queries:
  - Why or need for upload: Uploading your bank statement helps process your loan application faster and can lead to a better loan amount.
  - Which account: You need to upload the bank statement for your salary account, the account where your salary is credited.
  - Period of statement: You need to upload the bank statement for the last 4 months.
  - How to upload: There are several ways to upload your bank statement in the Zype app: through an Account Aggregator, using Netbanking, or by directly uploading a PDF file.
  - Where to upload: All these upload options are available within the Zype app.
  - When to upload: Upload the salary bank statement as soon as possible so that we can process your application faster.
  - Facing issues while uploading or failed while uploading: Our customer support executive will contact you shortly.
  - Requesting callback: Our customer support executive will contact you shortly.
  - For any other queries: Our customer support executive will contact you shortly.
- Responses MUST be in polite, conversational English, ending with a period (.). Aim for 1 short sentences directly to the point.
- Do not repeat the user's query in your response.
- After addressing the query, if the user seems ready (e.g., "Okay", "Alright"), ask for an ETA if not provided. Example: "Okay. So, will you be able to do it by this evening?"
- If the user provides an ETA (e.g., "I'll do it today"), acknowledge and close. Example: "That's great. We'll wait for your statement. Thank you! Have a good day."
- If the user says they have already uploaded, thank them and
 close. Example: "Thanks for uploading, we'll start processing your application. Have a good day."
- If the user says they have taken a loan from somewhere else, thank them for informing and tell them that they can still upload the statement and get an offer if they want a loan in future and end. Example: "Thanks for the update. You can still share your bank statement to check future offers. Thank you! Have a good day."
- If the user says they dont want a loan anymore, acknowledge it and politely ask for the reason for not wanting a loan. Example: "i respect your decision but may i ask you if there is a reason for not wanting a loan. We might offer better terms or resolve your concern"
- If the user says they have a preapproved offer and why ask for additional documents, acknowledge and tell them that the offer was based on initial checks and we need a few documents verify latest details and finalize it. Example: "Your offer is based on initial checks. We need a few documents to verify latest details and finalize it."
- If the user says they don’t have 4 months statement or they have less than 4 months statement, acknowledge and tell them that 4 months bank statement is required to verify the income and process the loan, if they are not sure how to download it we can arrange a call back to guide you. Example: "I can understand, but the 4-month bank statement is required to verify the income and process the loan smoothly, if you are not sure how to download it we can arrange a call back to guide you."
- If the user says their phone number isn’t linked to bank for AA (Account Aggregator), acknowledge and tell them to link their number at the bank or upload their 4-month statement as an alternative. Example: "Please link your number at the bank or upload your 4-month statement as an alternative."

**Strict Rules**:
- NO "Is there anything else...", NO PII requests, NO financial advice, NO arguing.
- If abusive, disengage: "I'm unable to continue this conversation. Thank you, have a good day."
- Preferred closing: "Thank you! Have a good day."
"""

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
        print(f"[USER] EN Text: '{english_transcript}', Detected Lang: {detected_lang}")
        if not english_transcript.strip():
            return "", DEFAULT_USER_LANGUAGE
        words = english_transcript.split()
        if len(words) > 25:
            print(f"[USER] Transcript too long ({len(words)} words), rejecting")
            return "", detected_lang
        return english_transcript.strip(), detected_lang
    except Exception as e:
        print(f"[ERROR] STT-Translate failed: {e}")
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
            return translated_text
        else:
            print(f"[ERROR] Translate failed: Empty translated_text")
            return text_to_translate
    except Exception as e:
        print(f"[ERROR] Translate failed: {e}")
        return text_to_translate


def get_audio_from_text_with_language(english_text: str, target_language_code: str):
    """Generate audio from English text by translating and using TTS"""
    if not english_text or not english_text.strip():
        return np.zeros(int(AUDIO_SAMPLE_RATE * 0.1), dtype=np.int16), AUDIO_SAMPLE_RATE
    translated_text = translate_text_sarvam(english_text, target_language_code)
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
        return audio_data, sample_rate
    except Exception as e:
        print(f"[ERROR] TTS failed: {e}")
        if target_language_code != "en-IN":
            print("[ERROR] TTS Fallback: Trying English")
            return get_audio_from_text_with_language(english_text, "en-IN")
        return np.zeros(int(AUDIO_SAMPLE_RATE * 0.1), dtype=np.int16), AUDIO_SAMPLE_RATE


def post_process_llm_english_response(english_response: str, user_english_input: str):
    """Post-process LLM English response"""
    response = english_response.strip()
    if response.startswith("- System:") or response.startswith("System:"):
        response = response.split(":", 1)[1].strip()
    if response.startswith("Assistant:"):
        response = response[len("Assistant:"):].strip()
    response = response.replace('–', '')
    response = re.sub(r"'", '', response)
    if not response:
        return "You can upload your statement in the Zype app."
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
    prohibited_phrases_english = ["any other questions", "further assistance", "feel free to ask"]
    if any(phrase.lower() in final_response.lower() for phrase in prohibited_phrases_english):
        print(f"[LLM] Prohibited phrase detected, using fallback")
        return "You can upload your statement in the Zype app."
    return final_response.strip()


# --- Audio Generation Functions ---
def generate_ai_greeting_and_identity_check(customer_name: str) -> str:
    return f"Hello, am I speaking with {customer_name}?"


def generate_ai_self_introduction() -> str:
    return f"This is {AI_NAME} calling from {COMPANY_NAME_FOR_LLM}."


def generate_ai_main_purpose_statement():
    return "We've noticed that the salary bank statement for your loan application hasn't been uploaded yet. Please upload it as soon as possible."


# --- LLM Setup ---
llm = ChatOpenAI(
    openai_api_base=LLM_API_BASE_URL,
    openai_api_key="NotNeeded",
    model_name=CURRENT_MODEL_ID_FROM_SERVER,
    temperature=0.5,
    max_tokens=LLM_MAX_TOKENS,
)


# --- Chat History Setup ---
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return an in-memory chat history for the session"""
    return InMemoryChatMessageHistory()


# --- Enhanced Call Manager with Better VAD ---
class EnhancedCallManager:
    def __init__(self, customer_name="हीरा"):
        self.customer_name = customer_name
        self.session_id = "default_session"  # Single session for simplicity
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_instruction_english),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        self.runnable = self.prompt_template | llm
        self.conversation = RunnableWithMessageHistory(
            runnable=self.runnable,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        self.state = CallState.WAITING
        self.conversation_stage = "greeting"
        self.unclear_input_count = 0
        self.current_user_language = DEFAULT_USER_LANGUAGE
        self.language_locked = False

        # Enhanced recording state
        self.is_recording = False
        self.audio_buffer = []
        self.recording_start_time = None
        self.last_audio_time = None
        self.ai_speech_end_time = None

        # VAD instance
        self.vad = VoiceActivityDetector()

    def clear_audio_buffers(self):
        """Clear all audio buffers and reset VAD"""
        # print("[VAD] Clearing audio buffers")
        self.audio_buffer = []
        self.vad.reset()
        self.is_recording = False
        self.recording_start_time = None
        self.last_audio_time = None

    def start_ai_speaking(self):
        """Transition to AI speaking state and clear buffers"""
        # print("[VAD] AI starting to speak - clearing buffers")
        self.state = CallState.AI_SPEAKING
        self.clear_audio_buffers()

    def finish_ai_speaking(self):
        """Transition from AI speaking to post-delay state"""
        # print(f"[VAD] AI finished speaking - starting {POST_AI_SPEECH_DELAY}s delay")
        self.state = CallState.POST_AI_DELAY
        self.ai_speech_end_time = time.time()
        self.clear_audio_buffers()  # Clear any residual audio

    def can_start_listening(self):
        """Check if enough time has passed since AI finished speaking"""
        if self.state != CallState.POST_AI_DELAY:
            return False
        if self.ai_speech_end_time is None:
            return True
        return time.time() - self.ai_speech_end_time >= POST_AI_SPEECH_DELAY

    def start_recording(self):
        """Start recording user speech"""
        if not self.is_recording:
            # print("[VAD] Starting speech recording")
            self.is_recording = True
            # Include pre-speech buffer
            pre_speech_audio = self.vad.get_pre_speech_audio()
            self.audio_buffer = pre_speech_audio.copy()
            self.recording_start_time = time.time()
            self.last_audio_time = time.time()
            self.state = CallState.LISTENING

    def should_stop_recording(self):
        """Check if recording should stop based on duration and silence"""
        if not self.is_recording:
            return False
        current_time = time.time()
        recording_duration = current_time - self.recording_start_time
        silence_duration = current_time - self.last_audio_time

        if recording_duration >= FIXED_RECORDING_DURATION:
            print(f"[VAD] Max duration reached ({recording_duration:.1f}s)")
            return True
        if (recording_duration >= MIN_RECORDING_DURATION and
                silence_duration >= SILENCE_DURATION_TO_STOP):
            # print(f"[VAD] Silence timeout ({silence_duration:.1f}s)")
            return True
        return False

    def stop_recording(self):
        """Stop recording and return duration"""
        if self.is_recording:
            duration = time.time() - self.recording_start_time
            # print(f"[VAD] Stopped recording ({duration:.1f}s)")
            self.is_recording = False
            self.vad.clear_pre_speech_buffer()  # Clear pre-speech buffer
            return True
        return False


async def process_audio_chunk(audio_chunk, call_manager, websocket):
    """Enhanced audio processing with better VAD and buffer management"""
    current_time = time.time()

    # Handle post-AI speech delay
    if call_manager.state == CallState.POST_AI_DELAY:
        if call_manager.can_start_listening():
            # print("[VAD] Post-AI delay complete - ready to listen")
            call_manager.state = CallState.WAITING
        else:
            # Still in delay period, ignore audio
            return

    # Only process audio when waiting or actively listening
    if call_manager.state not in [CallState.WAITING, CallState.LISTENING]:
        return

    # Use VAD to detect voice activity
    has_voice = call_manager.vad.is_voice_activity(audio_chunk)

    # Start recording if voice detected and not already recording
    if has_voice and not call_manager.is_recording and call_manager.state == CallState.WAITING:
        call_manager.start_recording()

    # Continue recording if already started
    if call_manager.is_recording:
        call_manager.audio_buffer.append(audio_chunk)

        # Update last audio time if there's voice activity
        if has_voice:
            call_manager.last_audio_time = current_time

        # Check if recording should stop
        if call_manager.should_stop_recording():
            await process_recorded_audio(call_manager, websocket)


async def process_recorded_audio(call_manager, websocket):
    """Process the recorded audio buffer with enhanced validation"""
    call_manager.stop_recording()
    call_manager.state = CallState.PROCESSING
    user_english_input = ""
    detected_lang = DEFAULT_USER_LANGUAGE

    if call_manager.audio_buffer:
        try:
            combined_audio = np.concatenate(call_manager.audio_buffer)
            duration = len(combined_audio) / EXOTEL_SAMPLE_RATE

            if duration < MIN_RECORDING_DURATION:
                print(f"[VAD] Recording too short ({duration:.1f}s), ignoring")
                call_manager.state = CallState.WAITING
                return

            # Resample to 16kHz for STT
            audio_16k = signal.resample(
                combined_audio,
                int(len(combined_audio) * AUDIO_SAMPLE_RATE / EXOTEL_SAMPLE_RATE)
            )

            # print(f"[VAD] Processing {duration:.1f}s of audio")
            user_english_input, detected_lang = speech_to_english_text(audio_16k.astype(np.int16))

            # Lock language after first successful recognition in conversation stage
            if (not call_manager.language_locked and
                    call_manager.conversation_stage == "conversation" and
                    user_english_input.strip()):
                call_manager.current_user_language = detected_lang if detected_lang in ALLOWED_LANGUAGES else DEFAULT_USER_LANGUAGE
                call_manager.language_locked = True
                print(f"[CALL] Language locked to: {call_manager.current_user_language}")

        except Exception as e:
            print(f"[ERROR] Audio processing failed: {e}")

    # Handle empty or unclear input
    if not user_english_input or not user_english_input.strip():
        user_english_input = "User was silent or input was unclear."
        call_manager.unclear_input_count += 1
        print(f"[USER] No transcript (count: {call_manager.unclear_input_count})")

    # Route to appropriate handler
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

    call_manager.conversation_stage = "introduction"
    await say_to_user(websocket, generate_ai_self_introduction(), call_manager)
    await asyncio.sleep(0.1)
    purpose = generate_ai_main_purpose_statement()
    await say_to_user(websocket, purpose, call_manager)
    call_manager.conversation_stage = "conversation"


async def handle_conversation(user_english_input, call_manager, websocket):
    """Handle main conversation with LLM"""
    if call_manager.unclear_input_count >= 4:
        await say_and_end(websocket,
                          "It seems there might be an issue with the line. We'll try to contact you later. Thank you.",
                          call_manager)
        return

    llm_input = user_english_input
    if len(llm_input) > 100:
        print(f"[LLM] Input: '{llm_input[:100]}...'")

    try:
        llm_response = await call_manager.conversation.ainvoke(
            {"input": llm_input},
            config={"configurable": {"session_id": call_manager.session_id}}
        )
        print(f"raw response: {llm_response.content}")
        llm_response_en = post_process_llm_english_response(llm_response.content, user_english_input)

        # Check for closing conditions
        closing_words = ["thank you! have a good day", "thanks for your time", "we'll try to contact you later", "contact you shortly"]
        is_closing = any(word in llm_response_en.lower() for word in closing_words)

        eta_keywords = ["today", "tomorrow", "evening", "morning", "will do", "i'll upload", "i will upload"]
        user_gave_eta = any(kw in user_english_input.lower() for kw in eta_keywords) and \
                        not any(
                            neg in user_english_input.lower() for neg in ["won't", "can't", "not able", "not today"])

        llm_acked_eta = ("we'll wait" in llm_response_en.lower() or "look forward" in llm_response_en.lower()) and \
                        ("thank you" in llm_response_en.lower() or "have a good day" in llm_response_en.lower())

        if user_gave_eta and not llm_acked_eta:
            eta_response = "That's great. We'll wait for your statement. Thank you! Have a good day."
            await say_and_end(websocket, eta_response, call_manager)
            return

        if is_closing or llm_acked_eta:
            print(f"[AI] Closing (EN): '{llm_response_en}'")
            await say_and_end(websocket, llm_response_en, call_manager)
        else:
            if llm_response_en != user_english_input:
                print(f"[LLM] Processed: '{llm_response_en}'")
            await say_to_user(websocket, llm_response_en, call_manager)

    except Exception as e:
        print(f"[ERROR] LLM failed: {e}")
        await say_and_end(websocket, "I'm facing a technical issue. We'll connect later. Thank you.", call_manager)


async def say_to_user(websocket, english_text, call_manager):
    """Say something in user's language and return to listening"""
    call_manager.start_ai_speaking()  # This clears buffers
    print(f"[AI] Speaking (EN): '{english_text}'")

    try:
        audio_data, rate = get_audio_from_text_with_language(english_text, call_manager.current_user_language)

        # Add to chat history
        history = get_session_history(call_manager.session_id)
        history.add_message(AIMessage(content=english_text))

        # Stream audio to user
        await stream_audio_to_exotel(websocket, audio_data, rate)

        # Finish speaking and start post-speech delay
        call_manager.finish_ai_speaking()

    except Exception as e:
        print(f"[ERROR] AI speech failed: {e}")
        call_manager.state = CallState.WAITING


async def say_and_end(websocket, english_text, call_manager):
    """Say something in user's language and end the call"""
    call_manager.start_ai_speaking()  # This clears buffers
    print(f"[AI] Closing (EN): '{english_text}'")

    try:
        audio_data, rate = get_audio_from_text_with_language(english_text, call_manager.current_user_language)

        # Add to chat history
        history = get_session_history(call_manager.session_id)
        history.add_message(AIMessage(content=english_text))

        # Stream audio to user
        await stream_audio_to_exotel(websocket, audio_data, rate)

    except Exception as e:
        print(f"[ERROR] AI speech failed: {e}")

    finally:
        call_manager.state = CallState.ENDED
        await websocket.close()
        print("[CALL] WebSocket connection closed to end the call")

async def stream_audio_to_exotel(websocket, audio_data, sample_rate):
    """Stream audio data to Exotel"""
    if sample_rate != EXOTEL_SAMPLE_RATE:
        audio_data_8k = signal.resample(
            audio_data,
            int(len(audio_data) * EXOTEL_SAMPLE_RATE / sample_rate)
        )
    else:
        audio_data_8k = audio_data

    audio_data_8k = audio_data_8k.astype(np.int16)
    chunk_size = CHUNK_SIZE

    for i in range(0, len(audio_data_8k), chunk_size):
        chunk = audio_data_8k[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')

        audio_bytes = chunk.tobytes()
        payload = base64.b64encode(audio_bytes).decode('utf-8')

        try:
            await websocket.send(json.dumps({
                'event': 'media',
                'media': {'payload': payload}
            }))
            await asyncio.sleep(0.02)  # 20ms delay between chunks
        except Exception as e:
            print(f"[ERROR] Stream audio failed: {e}")
            break


# --- Main WebSocket Handler ---
async def handle_exotel_call(websocket):
    """Handle Exotel WebSocket connection"""
    print("[CALL] New call connected")
    call_manager = EnhancedCallManager()

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                event_type = data.get('event')

                if event_type == 'connect':
                    print("[CALL] Connected")
                elif event_type == 'start':
                    print("[CALL] Call started")
                    await asyncio.sleep(0.3)
                    await say_to_user(
                        websocket,
                        generate_ai_greeting_and_identity_check(call_manager.customer_name),
                        call_manager
                    )
                elif event_type == 'media':
                    # Only process media when not AI speaking or processing
                    if call_manager.state not in [CallState.AI_SPEAKING, CallState.PROCESSING, CallState.ENDED]:
                        payload = data.get('media', {}).get('payload', '')
                        if payload:
                            try:
                                audio_data = base64.b64decode(payload)
                                audio_chunk = np.frombuffer(audio_data, dtype=np.int16)
                                await process_audio_chunk(audio_chunk, call_manager, websocket)
                            except Exception as e:
                                print(f"[ERROR] Media processing failed: {e}")
                elif event_type == 'stop':
                    print("[CALL] Call stopped")
                    break

            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decode failed: {e}")
            except Exception as e:
                print(f"[ERROR] Message processing failed: {e}")

    except websockets.exceptions.ConnectionClosed:
        print("[CALL] Connection closed")
    except Exception as e:
        print(f"[ERROR] Call failed: {e}")
    finally:
        print("[CALL] Call ended")


async def main():
    """Start the server"""
    print(f"[CALL] Starting server on ws://localhost:{WEBSOCKET_PORT}")

    async with websockets.serve(handle_exotel_call, "localhost", WEBSOCKET_PORT):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[CALL] Server stopped")
    except Exception as e:
        print(f"[ERROR] Server failed: {e}")
        traceback.print_exc()