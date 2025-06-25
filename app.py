import streamlit as st
import httpx
import json
import base64
import google.generativeai as genai
import googlemaps
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import time

# --- Configuration ---
try:
    ASTRO_API_USER_ID = st.secrets["ASTRO_API_USER_ID"]
    ASTRO_API_KEY = st.secrets["ASTRO_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    COLLECTION_NAME = "pandit_books"
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError as e:
    st.error(f"🚨 Missing Secret for '{e.args[0]}'.")
    st.stop()

# --- Services Initialization ---
@st.cache_resource
def get_services():
    llm_model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
    gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    return llm_model, gmaps_client, qdrant_client, embedding_model

llm_model, gmaps_client, qdrant_client, embedding_model = get_services()

# --- Core Logic Functions ---
def filter_user_question(question: str):
    prompt = f"""Analyze the user's question and classify it into "ASTROLOGY" or "META". User question: "{question}" -> Category:"""
    try:
        response = llm_model.generate_content(prompt)
        return "ASTROLOGY" if "ASTROLOGY" in response.text.strip().upper() else "META"
    except Exception: return "ASTROLOGY"

def get_final_answer(retrieved_context: str, kundli_json: dict, question: str, chat_history: str):
    """
    This is the single, powerful function that generates the final, accurate answer.
    It synthesizes the user's real chart data with retrieved astrological principles.
    """
    kundli_summary = json.dumps(kundli_json, indent=2)
    greeting_instruction = "- **Greeting:** Do not use any greeting."
    if not st.session_state.get("greeting_sent", False):
        greeting_instruction = "- **Greeting:** Begin with 'Namaste.' ONLY for this first message."
        st.session_state.greeting_sent = True

    # --- THE NEW, SIMPLIFIED, AND ROBUST PROMPT ---
    prompt = f"""You are Pandit 2.0, an expert Vedic Astrologer. Your task is to synthesize information from two sources to create a final, wise, and impactful answer for the user.

    **--- YOUR DATA SOURCES ---**
    1.  **The User's Chart Data:** This is the absolute and ONLY source of truth for the user's personal details, Dasha periods, and planetary placements.
    2.  **Retrieved Astrological Principles:** This is a knowledge base of timeless astrological rules.

    **--- YOUR NON-NEGOTIABLE CORE RULES ---**
    1.  **NEVER USE EXAMPLES FROM PRINCIPLES:** The `Retrieved Principles` may contain examples from other people's charts (e.g., "a person got a job in 1995"). You **MUST** strictly ignore these. They are not the user's life. Any specific life event, date, or personal detail you mention **MUST** come from the `User's Chart Data` ONLY.
    2.  **PRIORITIZE DASHA FOR TIMING:** Timing questions (e.g., "When will I get a job?") **MUST** be answered primarily using the Mahadasha and Antardasha lords provided in the `User's Chart Data`.
    3.  **JUSTIFY YOUR ANALYSIS:** You must justify your answer by mentioning specific planets (graha), houses (bhav), Dasha lords, and signs (rashi) from the `User's Chart Data`.
    4.  **MATCH THE LANGUAGE:** Reply in the same language as the "Latest User Question".
    5.  **PERSONA & GREETING:** Your tone is wise and confident. {greeting_instruction} Aim for two well-structured, explanatory paragraphs.

    ---
    **SOURCE 1: RETRIEVED ASTROLOGICAL PRINCIPLES (For knowledge only)**
    {retrieved_context}
    ---
    **SOURCE 2: THE USER'S CHART DATA (The only source of personal truth)**
    - *Chart & Dasha Data:* {kundli_summary}
    - *Conversation History:* {chat_history}
    - *Latest User Question:* "{question}"

    **PANDIT 2.0'S FINAL, PERSONALIZED, AND ACCURATE RESPONSE:**
    """
    try:
        response_stream = llm_model.generate_content(prompt, stream=True)
        for chunk in response_stream:
            yield chunk.text
    except Exception as e: yield f"Error: {e}"

@st.cache_data
def get_coordinates(_gmaps_client, city_name: str):
    try:
        geocode_result = _gmaps_client.geocode(city_name)
        return geocode_result[0]['geometry']['location']['lat'], geocode_result[0]['geometry']['location']['lng'] if geocode_result else (None, None)
    except Exception: return None, None

def get_kundli_and_charts(day, month, year, hour, minute, lat, lon, tzone):
    all_data = {}
    try:
        payload = {"day": day, "month": month, "year": year, "hour": hour, "min": minute, "lat": lat, "lon": lon, "tzone": tzone}
        auth_string = f"{ASTRO_API_USER_ID}:{ASTRO_API_KEY}"
        headers = {"Authorization": f"Basic {base64.b64encode(auth_string.encode()).decode()}"}
        
        with httpx.Client(timeout=45.0) as client:
            st.write("Fetching birth details...")
            all_data['kundli_data'] = client.post("https://json.astrologyapi.com/v1/astro_details", json=payload, headers=headers).raise_for_status().json()
            st.write("Fetching Major Dasha periods...")
            all_data['major_vdasha'] = client.post("https://json.astrologyapi.com/v1/major_vdasha", json=payload, headers=headers).raise_for_status().json()
            st.write("Fetching Current Dasha...")
            all_data['current_vdasha'] = client.post("https://json.astrologyapi.com/v1/current_vdasha", json=payload, headers=headers).raise_for_status().json()
            all_data['charts'] = {}
            for key, chart_id in {"d1_svg": "D1", "d9_svg": "D9", "moon_svg": "MOON", "chalit_svg": "chalit"}.items():
                st.write(f"Generating {chart_id} chart...")
                chart_payload = {**payload, "chart_style": "NORTH_INDIAN"}
                resp = client.post(f"https://json.astrologyapi.com/v1/horo_chart_image/{chart_id}", json=chart_payload, headers=headers).raise_for_status()
                try:
                    data = resp.json()
                    if isinstance(data, dict): all_data['charts'][key] = data.get('svg')
                    else: all_data['charts'][key] = None
                except json.JSONDecodeError:
                    raw_text = resp.text
                    if raw_text.strip().lower().startswith('<svg'): all_data['charts'][key] = raw_text
                    else: all_data['charts'][key] = None
        return all_data
    except Exception as e:
        st.error(f"Failed to fetch Astrology API data: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="Pandit 2.0 Pro", layout="centered")
st.title("✨ Pandit 2.0 - AI Astrologer")

if "kundli_data" not in st.session_state: st.session_state.kundli_data = None
if "messages" not in st.session_state: st.session_state.messages = []
if "greeting_sent" not in st.session_state: st.session_state.greeting_sent = False

with st.sidebar:
    st.header("Enter Birth Details")
    day, month, year = st.number_input("Day", 1, 31, 15), st.number_input("Month", 1, 12, 5), st.number_input("Year", 1990, 2023, 2001)
    hour, minute = st.number_input("Hour (24h)", 0, 23, 14), st.number_input("Minute", 0, 59, 30)
    tzone, city_name = st.number_input("Timezone", value=5.5, format="%.1f"), st.text_input("City of Birth", "New Delhi, India")

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
                with st.spinner("Pandit is analyzing..."):
                    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                    query_vector = embedding_model.encode(prompt).tolist()
                    search_results = qdrant_client.search(collection_name=COLLECTION_NAME, query_vector=query_vector, limit=5)
                    retrieved_context = "\n\n---\n\n".join([res.payload['text_chunk'] for res in search_results])
                    
                    # --- CALLING THE NEW, SIMPLIFIED FUNCTION ---
                    response_generator = get_final_answer(retrieved_context, st.session_state.kundli_data, prompt, history_str)
                    response = st.write_stream(response_generator)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("👋 Welcome! Please provide your birth details in the sidebar to begin.")
