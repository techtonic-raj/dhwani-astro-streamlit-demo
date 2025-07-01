# import streamlit as st
# import httpx
# import json
# import base64
# import google.generativeai as genai
# import googlemaps
# from qdrant_client import QdrantClient
# from sentence_transformers import SentenceTransformer
# import time

# # --- Configuration ---
# try:
#     ASTRO_API_USER_ID = st.secrets["ASTRO_API_USER_ID"]
#     ASTRO_API_KEY = st.secrets["ASTRO_API_KEY"]
#     GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
#     GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]
#     QDRANT_URL = st.secrets["QDRANT_URL"]
#     QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
#     COLLECTION_NAME = "pandit_books"
#     genai.configure(api_key=GEMINI_API_KEY)
# except KeyError as e:
#     st.error(f"🚨 Missing Secret for '{e.args[0]}'.")
#     st.stop()

# # --- Services Initialization ---
# @st.cache_resource
# def get_services():
#     llm_model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
#     gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
#     qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
#     embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
#     return llm_model, gmaps_client, qdrant_client, embedding_model

# llm_model, gmaps_client, qdrant_client, embedding_model = get_services()

# # --- Core Logic Functions ---
# def filter_user_question(question: str):
#     prompt = f"""Analyze the user's question and classify it into "ASTROLOGY" or "META". User question: "{question}" -> Category:"""
#     try:
#         response = llm_model.generate_content(prompt)
#         return "ASTROLOGY" if "ASTROLOGY" in response.text.strip().upper() else "META"
#     except Exception: return "ASTROLOGY"

# def get_final_answer(retrieved_context: str, kundli_json: dict, question: str, chat_history: str):
#     """
#     This is the single, powerful function that generates the final, accurate answer.
#     It synthesizes the user's real chart data with retrieved astrological principles.
#     """
#     kundli_summary = json.dumps(kundli_json, indent=2)
#     greeting_instruction = "- **Greeting:** Do not use any greeting."
#     if not st.session_state.get("greeting_sent", False):
#         greeting_instruction = "- **Greeting:** Begin with 'Namaste.' ONLY for this first message."
#         st.session_state.greeting_sent = True

#     # --- THE NEW, SIMPLIFIED, AND ROBUST PROMPT ---
#     prompt = f"""You are Pandit 2.0, an expert Vedic Astrologer. Your task is to synthesize information from two sources to create a final, wise, and impactful answer for the user.

#     **--- YOUR DATA SOURCES ---**
#     1.  **The User's Chart Data:** This is the absolute and ONLY source of truth for the user's personal details, Dasha periods, and planetary placements.
#     2.  **Retrieved Astrological Principles:** This is a knowledge base of timeless astrological rules.

#     **--- YOUR NON-NEGOTIABLE CORE RULES ---**
#     1.  **NEVER USE EXAMPLES FROM PRINCIPLES:** The `Retrieved Principles` may contain examples from other people's charts (e.g., "a person got a job in 1995"). You **MUST** strictly ignore these. They are not the user's life. Any specific life event, date, or personal detail you mention **MUST** come from the `User's Chart Data` ONLY.
#     2.  **PRIORITIZE DASHA FOR TIMING:** Timing questions (e.g., "When will I get a job?") **MUST** be answered primarily using the Mahadasha and Antardasha lords provided in the `User's Chart Data`.
#     3.  **JUSTIFY YOUR ANALYSIS:** You must justify your answer by mentioning specific planets (graha), houses (bhav), Dasha lords, and signs (rashi) from the `User's Chart Data`.
#     4.  **MATCH THE LANGUAGE:** Reply in the same language as the "Latest User Question".
#     5.  **PERSONA & GREETING:** Your tone is wise and confident. {greeting_instruction} Aim for two well-structured, explanatory paragraphs.

#     ---
#     **SOURCE 1: RETRIEVED ASTROLOGICAL PRINCIPLES (For knowledge only)**
#     {retrieved_context}
#     ---
#     **SOURCE 2: THE USER'S CHART DATA (The only source of personal truth)**
#     - *Chart & Dasha Data:* {kundli_summary}
#     - *Conversation History:* {chat_history}
#     - *Latest User Question:* "{question}"

#     **PANDIT 2.0'S FINAL, PERSONALIZED, AND ACCURATE RESPONSE:**
#     """
#     try:
#         response_stream = llm_model.generate_content(prompt, stream=True)
#         for chunk in response_stream:
#             yield chunk.text
#     except Exception as e: yield f"Error: {e}"

# @st.cache_data
# def get_coordinates(_gmaps_client, city_name: str):
#     try:
#         geocode_result = _gmaps_client.geocode(city_name)
#         return geocode_result[0]['geometry']['location']['lat'], geocode_result[0]['geometry']['location']['lng'] if geocode_result else (None, None)
#     except Exception: return None, None

# def get_kundli_and_charts(day, month, year, hour, minute, lat, lon, tzone):
#     all_data = {}
#     try:
#         payload = {"day": day, "month": month, "year": year, "hour": hour, "min": minute, "lat": lat, "lon": lon, "tzone": tzone}
#         auth_string = f"{ASTRO_API_USER_ID}:{ASTRO_API_KEY}"
#         headers = {"Authorization": f"Basic {base64.b64encode(auth_string.encode()).decode()}"}
        
#         with httpx.Client(timeout=45.0) as client:
#             st.write("Fetching birth details...")
#             all_data['kundli_data'] = client.post("https://json.astrologyapi.com/v1/astro_details", json=payload, headers=headers).raise_for_status().json()
#             st.write("Fetching Major Dasha periods...")
#             all_data['major_vdasha'] = client.post("https://json.astrologyapi.com/v1/major_vdasha", json=payload, headers=headers).raise_for_status().json()
#             st.write("Fetching Current Dasha...")
#             all_data['current_vdasha'] = client.post("https://json.astrologyapi.com/v1/current_vdasha", json=payload, headers=headers).raise_for_status().json()
#             all_data['charts'] = {}
#             for key, chart_id in {"d1_svg": "D1", "d9_svg": "D9", "moon_svg": "MOON", "chalit_svg": "chalit"}.items():
#                 st.write(f"Generating {chart_id} chart...")
#                 chart_payload = {**payload, "chart_style": "NORTH_INDIAN"}
#                 resp = client.post(f"https://json.astrologyapi.com/v1/horo_chart_image/{chart_id}", json=chart_payload, headers=headers).raise_for_status()
#                 try:
#                     data = resp.json()
#                     if isinstance(data, dict): all_data['charts'][key] = data.get('svg')
#                     else: all_data['charts'][key] = None
#                 except json.JSONDecodeError:
#                     raw_text = resp.text
#                     if raw_text.strip().lower().startswith('<svg'): all_data['charts'][key] = raw_text
#                     else: all_data['charts'][key] = None
#         return all_data
#     except Exception as e:
#         st.error(f"Failed to fetch Astrology API data: {e}")
#         return None

# # --- Streamlit UI ---
# st.set_page_config(page_title="Pandit 2.0 Pro", layout="centered")
# st.title("✨ Pandit 2.0 - AI Astrologer")

# if "kundli_data" not in st.session_state: st.session_state.kundli_data = None
# if "messages" not in st.session_state: st.session_state.messages = []
# if "greeting_sent" not in st.session_state: st.session_state.greeting_sent = False

# with st.sidebar:
#     st.header("Enter Birth Details")
#     day, month, year = st.number_input("Day", 1, 31, 15), st.number_input("Month", 1, 12, 5), st.number_input("Year", 1990, 2023, 2001)
#     hour, minute = st.number_input("Hour (24h)", 0, 23, 14), st.number_input("Minute", 0, 59, 30)
#     tzone, city_name = st.number_input("Timezone", value=5.5, format="%.1f"), st.text_input("City of Birth", "New Delhi, India")

#     if st.button("Generate & Analyze Kundli", type="primary"):
#         lat, lon = get_coordinates(gmaps_client, city_name)
#         if lat and lon:
#             with st.spinner("Calculating your cosmic blueprint..."):
#                 st.session_state.kundli_data = get_kundli_and_charts(day, month, year, hour, minute, lat, lon, tzone)
#                 st.session_state.messages, st.session_state.greeting_sent = [], False
#                 if st.session_state.kundli_data:
#                     st.session_state.messages.append({"role": "assistant", "content": "Your charts and Dasha periods are ready. Ask your question."})
#                     st.rerun()
#         else: st.error(f"Could not find coordinates for '{city_name}'.")

# if st.session_state.kundli_data:
#     with st.expander("View Your Astrological Charts", expanded=False):
#         charts = st.session_state.kundli_data.get('charts', {})
#         cols = st.columns(2)
#         for i, (key, name) in enumerate([("d1_svg", "Lagna (D1)"), ("d9_svg", "Navamsa (D9)"), ("moon_svg", "Moon Chart"), ("chalit_svg", "Chalit Chart")]):
#             with cols[i % 2]:
#                 st.subheader(name)
#                 if img := charts.get(key): st.image(img)
#                 else: st.warning(f"{name} not available.")
    
#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]): st.markdown(msg["content"], unsafe_allow_html=True)

#     if prompt := st.chat_input("Ask about your destiny..."):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"): st.markdown(prompt)

#         with st.chat_message("assistant"):
#             if filter_user_question(prompt) == "META":
#                 response = "I am Pandit 2.0, an AI Vedic Astrologer. Please ask questions about your chart."
#                 st.write(response)
#             else:
#                 with st.spinner("Pandit is analyzing..."):
#                     history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
#                     query_vector = embedding_model.encode(prompt).tolist()
#                     search_results = qdrant_client.search(collection_name=COLLECTION_NAME, query_vector=query_vector, limit=5)
#                     retrieved_context = "\n\n---\n\n".join([res.payload['text_chunk'] for res in search_results])
                    
#                     # --- CALLING THE NEW, SIMPLIFIED FUNCTION ---
#                     response_generator = get_final_answer(retrieved_context, st.session_state.kundli_data, prompt, history_str)
#                     response = st.write_stream(response_generator)
#             st.session_state.messages.append({"role": "assistant", "content": response})
# else:
#     st.info("👋 Welcome! Please provide your birth details in the sidebar to begin.")



import streamlit as st
import httpx
import json
import base64
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import googlemaps
import time

# --- ----------------------------------------------------------------- ---
# --- ⚙️ 1. SECURE CONFIGURATION (Reads from .streamlit/secrets.toml) ⚙️ ---
# --- ----------------------------------------------------------------- ---
try:
    # Your existing secrets
    ASTRO_API_USER_ID = st.secrets["ASTRO_API_USER_ID"]
    ASTRO_API_KEY = st.secrets["ASTRO_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]
    
    genai.configure(api_key=GEMINI_API_KEY)

except KeyError as e:
    st.error(f"🚨 Missing Secret! Please check your secrets file for '{e.args[0]}'.")
    st.stop()


# --- -------------------------------------------------- ---
# --- 🧠 2. INITIALIZE AI MODELS & SERVICES (Cached) 🧠 ---
# --- -------------------------------------------------- ---
@st.cache_resource
def get_services():
    """Initializes the THREE different AI models and the Google Maps client."""
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    model_name = st.secrets["GEMINI_MODEL"]

    # Agent 1: The Guard (Fast & Cheap, for classification)
    filter_model = genai.GenerativeModel(model_name, safety_settings=safety_settings)

    # Agent 2: The Analyst (Powerful, no search tools now)
    analyst_model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
    
    # Agent 3: The Orator (Fast & Persona-driven)
    orator_model = genai.GenerativeModel(model_name, safety_settings=safety_settings)

    gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    
    return filter_model, analyst_model, orator_model, gmaps_client


filter_model, analyst_model, orator_model, gmaps_client = get_services()


# --- -------------------------------------------- ---
# --- 📞 3. CORE LOGIC & API FUNCTIONS 📞 ---
# --- -------------------------------------------- ---

def style_svg_chart(svg_string: str, line_color: str = "#FCA311", text_color: str = "#E0E0E0"):
    """Takes a raw SVG string and replaces black strokes and fills with vibrant colors."""
    if not svg_string or not svg_string.strip().lower().startswith('<svg'):
        return None # Return None if the input is not a valid SVG string
    
    # Replace black strokes (lines) with a vibrant gold/yellow
    svg_string = svg_string.replace('stroke="black"', f'stroke="{line_color}"')
    svg_string = svg_string.replace('stroke="#000000"', f'stroke="{line_color}"')
    
    # Replace black fills (text) with a light grey/white
    svg_string = svg_string.replace('fill="black"', f'fill="{text_color}"')
    svg_string = svg_string.replace('fill="#000000"', f'fill="{text_color}"')
    
    return svg_string

@st.cache_data
def get_coordinates(_gmaps_client, city_name: str):
    try:
        geocode_result = _gmaps_client.geocode(city_name)
        if geocode_result: return geocode_result[0]['geometry']['location']['lat'], geocode_result[0]['geometry']['location']['lng']
    except Exception: return None, None
    return None, None

def get_kundli_and_charts(day, month, year, hour, minute, lat, lon, tzone):
    all_data = {}
    try:
        payload = {"day": day, "month": month, "year": year, "hour": hour, "min": minute, "lat": lat, "lon": lon, "tzone": tzone}
        auth_string = f"{ASTRO_API_USER_ID}:{ASTRO_API_KEY}"
        headers = {"Authorization": f"Basic {base64.b64encode(auth_string.encode()).decode()}"}
        
        with httpx.Client(timeout=45.0) as client:
            st.write("Fetching birth details...")
            all_data['kundli_data'] = client.post(f"{ASTRO_API_BASE_URL}/astro_details", json=payload, headers=headers).raise_for_status().json()
            st.write("Fetching Dasha periods...")
            all_data['major_vdasha'] = client.post(f"{ASTRO_API_BASE_URL}/major_vdasha", json=payload, headers=headers).raise_for_status().json()
            
            all_data['charts'] = {}
            for key, chart_id in {"d1_svg": "D1", "d9_svg": "D9", "moon_svg": "MOON", "chalit_svg": "chalit"}.items():
                st.write(f"Generating {chart_id} chart...")
                chart_payload = {**payload, "chart_style": "NORTH_INDIAN"}
                resp = client.post(f"{ASTRO_API_BASE_URL}/horo_chart_image/{chart_id}", json=chart_payload, headers=headers).raise_for_status()
                
                raw_svg = None
                try:
                    data = resp.json()
                    if isinstance(data, dict): raw_svg = data.get('svg')
                except json.JSONDecodeError:
                    raw_text = resp.text
                    if raw_text.strip().lower().startswith('<svg'): raw_svg = raw_text
                
                all_data['charts'][key] = style_svg_chart(raw_svg)
        return all_data
    except Exception as e:
        st.error(f"Failed to fetch Astrology API data: {e}")
        return None

def run_ai_pipeline(kundli_json: dict, question: str, chat_history: str):
    """This function runs the full Three-Agent Pipeline."""
    guard_prompt = f"""Analyze the user's question. Classify it as 'valid_astrology', 'greeting', or 'off_topic'. Respond with ONLY one of these words. User Question: "{question}" """
    try:
        guard_response = guard_model.generate_content(guard_prompt)
        classification = guard_response.text.strip().lower()
    except Exception as e:
        st.warning(f"Filter model failed, proceeding with caution. Error: {e}")
        classification = 'valid_astrology' # Default to valid if the guard fails

    if 'off_topic' in classification:
        yield "Namaste. I am Pandit 2.0, your personal astrological guide. My purpose is to provide insights based on your birth chart. Please feel free to ask me any questions about your life's path."
        return
    if 'greeting' in classification:
        yield "Namaste. How may I assist you with your astrological queries today?"
        return

    analyst_prompt = f"""You are a master data analyst AI. Your task is to synthesize all available information into a detailed, factual analysis.
    1. Analyze the User's Birth Chart Data.
    2. Analyze the Conversation History for context.
    3. Combine everything into a detailed, fact-rich, but unformatted block of text that answers the user's question. Just provide the core analysis.

    **USER'S BIRTH CHART DATA:** {json.dumps(kundli_json)}
    **CONVERSATION HISTORY:** {chat_history}
    **USER QUESTION:** "{question}"
    **DETAILED FACTUAL ANALYSIS:**"""
    
    try:
        analysis_text = analyst_model.generate_content(analyst_prompt).text
    except Exception as e:
        yield f"Error during analysis phase: {e}"; return

    greeting_instruction = "" if st.session_state.greeting_sent else "- Greeting: Begin with 'Namaste.' ONLY for this first message."
    if not st.session_state.greeting_sent: st.session_state.greeting_sent = True

    orator_prompt = f"""You are Pandit 2.0, a world-class Vedic Astrologer. Your task is to take the provided 'RAW ANALYSIS' and rewrite it into a final, polished response.
    **Your Persona & Structure (CRUCIAL):**
    - Tone: Wise, confident, and empathetic.
    - Highlighting: Use **bold text** for key astrological terms (planets, houses, signs, yogas).
    - Concise & Structured: Aim for two well-structured, easy-to-read paragraphs. {greeting_instruction}
    **RAW ANALYSIS TO REWRITE:** {analysis_text}
    **PANDIT 2.0'S POLISHED RESPONSE:**"""
    
    try:
        response_stream = orator_model.generate_content(orator_prompt, stream=True)
        for chunk in response_stream:
            yield chunk.text
    except Exception as e:
        yield f"Error during formatting phase: {e}"

# --- --------------------------------------- ---
# --- 🎨 4. THE STREAMLIT USER INTERFACE 🎨 ---
# --- --------------------------------------- ---
st.set_page_config(page_title="Pandit 2.0 Pro", layout="centered")
st.title("✨ Pandit 2.0 - AI Astrologer")


# --- THIS IS THE NEW CODE BLOCK TO ADD ---
# Inject custom CSS to make the SVG charts colorful and visible on a dark theme.
CHART_STYLE = """
<style>
    /* Target the container for each chart image */
    div[data-testid="stImage"] {
        background-color: #1a1a2e; /* A dark, celestial blue background */
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #4a4e69;
    }

    /* Target the elements *inside* the SVG provided by the API */
    div[data-testid="stImage"] svg text {
        fill: #E0E0E0 !important; /* Force all text (planets, numbers) to be a light grey/white */
        font-weight: bold;
        font-size: 14px;
        font-family: sans-serif;
    }
    
    div[data-testid="stImage"] svg path, 
    div[data-testid="stImage"] svg line,
    div[data-testid="stImage"] svg polyline {
        stroke: #FCA311 !important; /* A vibrant, golden-orange for all chart lines */
        stroke-width: 1.5; /* Make lines slightly thicker and more visible */
    }
</style>
"""
st.markdown(CHART_STYLE, unsafe_allow_html=True)
# --- END OF NEW CODE BLOCK ---






if "kundli_data" not in st.session_state: st.session_state.kundli_data = None
if "messages" not in st.session_state: st.session_state.messages = []
if "greeting_sent" not in st.session_state: st.session_state.greeting_sent = False

with st.sidebar:
    st.header("Enter Birth Details")
    day = st.number_input("Day", 1, 31, 15)
    month = st.number_input("Month", 1, 12, 5)
    year = st.number_input("Year", 1990, 2023, 2001)
    hour = st.number_input("Hour (24h)", 0, 23, 14)
    minute = st.number_input("Minute", 0, 59, 30)
    tzone = st.number_input("Timezone", value=5.5, format="%.1f")
    city_name = st.text_input("City of Birth", "New Delhi, India")

    if st.button("Generate & Analyze Kundli", type="primary"):
        lat, lon = get_coordinates(gmaps_client, city_name)
        if lat and lon:
            with st.spinner("Calculating your cosmic blueprint..."):
                st.session_state.kundli_data = get_kundli_and_charts(day, month, year, hour, minute, lat, lon, tzone)
                st.session_state.messages, st.session_state.greeting_sent = [], False
                if st.session_state.kundli_data:
                    st.session_state.messages.append({"role": "assistant", "content": "Your charts and Dasha periods are ready. Ask your question."})
                    st.rerun()
        else: st.error(f"Could not find coordinates for '{city_name}'.")

if st.session_state.kundli_data:
    with st.expander("View Your Astrological Charts", expanded=False):
        charts = st.session_state.kundli_data.get('charts', {})
        cols = st.columns(2)
        for i, (key, name) in enumerate([("d1_svg", "Lagna (D1)"), ("d9_svg", "Navamsa (D9)"), ("moon_svg", "Moon Chart"), ("chalit_svg", "Chalit Chart")]):
            with cols[i % 2]:
                st.subheader(name)
                if img := charts.get(key): st.image(img)
                else: st.warning(f"{name} not available.")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Ask about your destiny..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            if filter_user_question(prompt) == "META":
                response = "I am Pandit 2.0, an AI Vedic Astrologer. Please ask questions about your chart."
                st.write(response)
            else:
                with st.spinner("Pandit is deeply analyzing the cosmos..."):
                    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                    
                    # --- CALLING THE NEW MASTER PIPELINE ---
                    response_generator = run_ai_pipeline(st.session_state.kundli_data, prompt, history_str)
                    response = st.write_stream(response_generator)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("👋 Welcome! Please provide your birth details in the sidebar to begin.")
