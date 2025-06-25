import streamlit as st
import httpx
import json
import base64
import google.generativeai as genai
import googlemaps
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import time

# --- ----------------------------------------------------------------- ---
# --- ⚙️ 1. SECURE CONFIGURATION (Reads from .streamlit/secrets.toml) ⚙️ ---
# --- ----------------------------------------------------------------- ---
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
    st.error(f"🚨 Missing Secret! Please check your .streamlit/secrets.toml file or Streamlit Cloud secrets for '{e.args[0]}'.")
    st.stop()


# --- -------------------------------------------------- ---
# --- 🧠 2. INITIALIZE AI MODELS & SERVICES (Cached) 🧠 ---
# --- -------------------------------------------------- ---
@st.cache_resource
def get_services():
    llm_model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
    gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    return llm_model, gmaps_client, qdrant_client, embedding_model

llm_model, gmaps_client, qdrant_client, embedding_model = get_services()


# --- -------------------------------------------- ---
# --- 📞 3. API & CORE LOGIC FUNCTIONS 📞 ---
# --- -------------------------------------------- ---
def filter_user_question(question: str):
    prompt = f"""Analyze the user's question and classify it into one of two categories: "ASTROLOGY" or "META".
    User question: "{question}" -> Category:"""
    try:
        response = llm_model.generate_content(prompt)
        category = response.text.strip().upper()
        return "ASTROLOGY" if "ASTROLOGY" in category else "META"
    except Exception:
        return "ASTROLOGY"

def get_initial_llm_draft(kundli_json: dict, question: str, chat_history: str):
    kundli_summary = json.dumps(kundli_json, indent=2)
    prompt = f"""You are a junior Vedic Astrologer. Provide a direct, preliminary analysis based *only* on the provided birth chart and question. Be factual and concise.
    *USER'S BIRTH CHART DATA:* {kundli_summary}
    *LATEST USER QUESTION:* "{question}"
    *JUNIOR PANDIT'S PRELIMINARY DRAFT:*"""
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception:
        return "Error generating draft."

def get_final_reviewed_answer(initial_draft: str, retrieved_context: str, kundli_json: dict, question: str, chat_history: str):
    kundli_summary = json.dumps(kundli_json, indent=2)

    greeting_instruction = "- **Greeting:** Do not use any greeting. Go straight to the answer."
    if not st.session_state.get("greeting_sent", False):
        greeting_instruction = "- **Greeting:** Begin your response with 'Namaste.' ONLY for this first message."
        st.session_state.greeting_sent = True

    # --- THE NEW, FLAW-FIXING PROMPT ---
    prompt = f"""You are Pandit 2.0, an expert Vedic Astrologer. Your task is to provide a final, wise, and impactful answer to the user.

    **--- YOUR CORE RULES ---**
    1.  **THE USER'S CHART IS THE ONLY TRUTH:** The `Chart Data` provided is the absolute source of truth for the user's life.
    2.  **PRINCIPLES, NOT EXAMPLES:** The `Ancient Verses` retrieved from the database are for understanding timeless astrological **principles and rules only**. They might contain examples from other people's charts. **You MUST IGNORE and NEVER MENTION any specific life events, dates, or personal details found in these verses.** Your entire analysis must be a unique synthesis of the principles applied *exclusively* to the user's personal `Chart Data`.
    3.  **MATCH THE LANGUAGE:** You MUST reply in the same language as the "Latest User Question".
    4.  **BE CONCISE BUT EXPLANATORY:** Aim for two well-structured paragraphs. The answer must be short but rich with meaning, mentioning specific planets (graha), houses (bhav), or signs (rashi) from the user's chart to justify your analysis.
    5.  **PERSONA & GREETING:** Your tone is wise and confident. {greeting_instruction}

    ---
    **INTERNAL CONTEXT FOR PRINCIPLES ONLY (IGNORE SPECIFIC EXAMPLES HERE):**
    - *Ancient Verses:* {retrieved_context}
    - *Junior's Draft:* {initial_draft}
    ---
    **THE ONLY SOURCE OF TRUTH FOR PERSONAL DATA:**
    - *Chart Data:* {kundli_summary}
    - *Conversation:* {chat_history}
    - *Latest User Question:* "{question}"

    **PANDIT 2.0'S FINAL, PERSONALIZED, AND EXPLANATORY RESPONSE:**
    """
    
    try:
        response_stream = llm_model.generate_content(prompt, stream=True)
        for chunk in response_stream:
            yield chunk.text
    except Exception as e:
        yield f"Error communicating with the Senior AI Brain: {e}"

@st.cache_data
def get_coordinates(_gmaps_client, city_name: str):
    try:
        geocode_result = _gmaps_client.geocode(city_name)
        if geocode_result:
            return geocode_result[0]['geometry']['location']['lat'], geocode_result[0]['geometry']['location']['lng']
    except Exception as e:
        st.error(f"Geocoding Error: {e}")
    return None, None

def get_kundli_and_charts(day, month, year, hour, minute, lat, lon, tzone):
    all_data = {}
    try:
        payload = {"day": day, "month": month, "year": year, "hour": hour, "min": minute, "lat": lat, "lon": lon, "tzone": tzone}
        auth_string = f"{ASTRO_API_USER_ID}:{ASTRO_API_KEY}"
        b64_auth = base64.b64encode(auth_string.encode()).decode()
        headers = {"Authorization": f"Basic {b64_auth}", "Content-Type": "application/json"}
        with httpx.Client(timeout=45.0) as client:
            st.write("Fetching birth details...")
            main_response = client.post(f"https://json.astrologyapi.com/v1/astro_details", json=payload, headers=headers)
            main_response.raise_for_status()
            all_data['kundli_data'] = main_response.json()
            all_data['charts'] = {}
            charts_to_fetch = {"d1_svg": "D1", "d9_svg": "D9", "moon_svg": "MOON", "chalit_svg": "chalit"}
            for key, chart_id in charts_to_fetch.items():
                st.write(f"Generating {chart_id} chart...")
                chart_payload = {**payload, "chart_style": "NORTH_INDIAN"}
                chart_response = client.post(f"https://json.astrologyapi.com/v1/horo_chart_image/{chart_id}", json=chart_payload, headers=headers)
                chart_response.raise_for_status()
                try:
                    data = chart_response.json()
                    if isinstance(data, dict) and 'svg' in data:
                        all_data['charts'][key] = data['svg']
                    else:
                        all_data['charts'][key] = None
                except json.JSONDecodeError:
                    all_data['charts'][key] = None
        return all_data
    except Exception as e:
        st.error(f"Failed to fetch Astrology API data: {e}")
        return None

# --- --------------------------------------- ---
# --- 🎨 4. THE STREAMLIT USER INTERFACE 🎨 ---
# --- --------------------------------------- ---
st.set_page_config(page_title="Pandit 2.0 Pro", layout="centered")
st.title("✨ Pandit 2.0 - AI Astrologer")

if "kundli_data" not in st.session_state: st.session_state.kundli_data = None
if "messages" not in st.session_state: st.session_state.messages = []
if "greeting_sent" not in st.session_state: st.session_state.greeting_sent = False

with st.sidebar:
    st.header("Enter Birth Details")
    day = st.number_input("Day", 1, 31, 15)
    month = st.number_input("Month", 1, 12, 5)
    year = st.number_input("Year", 1990, 2023, 2001) # Default to 2001
    hour = st.number_input("Hour (24h)", 0, 23, 14)
    minute = st.number_input("Minute", 0, 59, 30)
    tzone = st.number_input("Timezone", value=5.5, format="%.1f")
    city_name = st.text_input("City of Birth", "New Delhi, India")
    if st.button("Generate & Analyze Kundli", type="primary"):
        lat, lon = get_coordinates(gmaps_client, city_name)
        if lat and lon:
            with st.spinner("Calculating your cosmic blueprint..."):
                st.session_state.kundli_data = get_kundli_and_charts(day, month, year, hour, minute, lat, lon, tzone)
                st.session_state.messages = []
                st.session_state.greeting_sent = False
                if st.session_state.kundli_data:
                    initial_greeting = "Your charts have been generated. I have analyzed the core of your horoscope and am ready for your questions."
                    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
                    st.rerun()
        else:
            st.error(f"Could not find coordinates for '{city_name}'.")

if not st.session_state.kundli_data:
    st.info("👋 Welcome! Please provide your birth details in the sidebar to begin your personalized consultation.")
else:
    with st.expander("View Your Astrological Charts", expanded=False):
        chart_data = st.session_state.kundli_data.get('charts', {})
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Lagna (D1)")
            if chart_data.get('d1_svg'): st.image(chart_data['d1_svg'])
            else: st.warning("D1 Chart not available.")
        with col2:
            st.subheader("Navamsa (D9)")
            if chart_data.get('d9_svg'): st.image(chart_data['d9_svg'])
            else: st.warning("D9 Chart not available.")
        with col1:
            st.subheader("Moon Chart")
            if chart_data.get('moon_svg'): st.image(chart_data['moon_svg'])
            else: st.warning("Moon Chart not available.")
        with col2:
            st.subheader("Chalit Chart")
            if chart_data.get('chalit_svg'): st.image(chart_data['chalit_svg'])
            else: st.warning("Chalit Chart not available.")
    
    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Ask about your destiny..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            category = filter_user_question(prompt)
            
            if category == "META":
                canned_response = "I am Pandit 2.0, an AI assistant for Vedic Astrology. My purpose is to provide guidance based on your birth chart. Please ask questions related to your horoscope."
                st.write(canned_response)
                st.session_state.messages.append({"role": "assistant", "content": canned_response})
            
            else:
                main_kundli_data = st.session_state.kundli_data.get('kundli_data', {})
                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                
                with st.spinner("Pandit is analyzing your chart..."):
                    initial_draft = get_initial_llm_draft(main_kundli_data, prompt, history_str)

                with st.spinner("Consulting ancient texts..."):
                    query_vector = embedding_model.encode(prompt).tolist()
                    search_results = qdrant_client.search(collection_name=COLLECTION_NAME, query_vector=query_vector, limit=10)
                    retrieved_context = "\n\n---\n\n".join([res.payload['text_chunk'] for res in search_results])

                with st.spinner("Senior Pandit is reviewing and refining the answer..."):
                    response_generator = get_final_reviewed_answer(initial_draft, retrieved_context, main_kundli_data, prompt, history_str)
                    full_response = st.write_stream(response_generator)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
