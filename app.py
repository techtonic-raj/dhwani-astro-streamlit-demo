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
# --- ⚙️ 1. SECURE CONFIGURATION (Uses Streamlit Secrets) ⚙️ ---
# --- ----------------------------------------------------------------- ---

# This section tells the app to look for secrets provided by the Streamlit hosting environment.
# You will paste your actual keys into the Streamlit deployment settings, NOT here.

try:
    # Astrology API Credentials
    ASTRO_API_URL = st.secrets["ASTRO_API_URL"]
    ASTRO_API_USER_ID = st.secrets["ASTRO_API_USER_ID"]
    ASTRO_API_KEY = st.secrets["ASTRO_API_KEY"]

    # Google Services API Keys
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]

    # Qdrant Cloud Credentials
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    COLLECTION_NAME = "pandit_books"
    
    # Configure the Gemini client
    genai.configure(api_key=GEMINI_API_KEY)

except KeyError as e:
    # If a secret is missing, stop the app and show a helpful error
    st.error(f"🚨 Missing Secret! Please add '{e.args[0]}' to your Streamlit Cloud secrets.")
    st.stop()


# --- -------------------------------------------------- ---
# --- 🧠 2. INITIALIZE AI MODELS & SERVICES (Cached) 🧠 ---
# --- -------------------------------------------------- ---

@st.cache_resource
def get_services():
    """Initializes all the necessary clients and models safely."""
    llm_model = genai.GenerativeModel('gemini-1.5-pro-latest')
    gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    return llm_model, gmaps_client, qdrant_client, embedding_model

llm_model, gmaps_client, qdrant_client, embedding_model = get_services()

# --- -------------------------------------------- ---
# --- 📞 3. API & CORE LOGIC FUNCTIONS 📞 ---
# --- -------------------------------------------- ---

@st.cache_data
def get_coordinates(_gmaps_client, city_name: str):
    """Converts a city name to latitude and longitude using Google Maps API."""
    try:
        geocode_result = _gmaps_client.geocode(city_name)
        if geocode_result:
            return geocode_result[0]['geometry']['location']['lat'], geocode_result[0]['geometry']['location']['lng']
    except Exception as e:
        st.error(f"Geocoding Error: {e}")
    return None, None

def get_kundli_data(day, month, year, hour, minute, lat, lon, tzone):
    """Calls your external astrology API. **YOU MUST ADAPT THIS TO MATCH YOUR API.**"""
    try:
        payload = {"day": day, "month": month, "year": year, "hour": hour, "min": minute, "lat": lat, "lon": lon, "tzone": tzone, "charts_format": "svg"}
        auth_string = f"{ASTRO_API_USER_ID}:{ASTRO_API_KEY}"
        b64_auth = base64.b64encode(auth_string.encode()).decode()
        headers = {"Authorization": f"Basic {b64_auth}", "Content-Type": "application/json"}
        with httpx.Client(timeout=30.0) as client:
            response = client.post(ASTRO_API_URL, json=payload, headers=headers)
            response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch Kundli data: {e}")
        return None

def get_llm_analysis(kundli_json: dict, question: str, chat_history: str, retrieved_context: str):
    """Sends the complete context to the Gemini model and streams the response."""
    kundli_summary = json.dumps(kundli_json, indent=2)
    prompt = f"""You are Pandit 2.0, a world-class Vedic Astrologer.

    **Your Persona:**
    - **Tone:** Wise, confident, empathetic, and direct.
    - **Greetings:** Always begin every response with a respectful greeting.

    **Your Answer Structure (CRUCIAL):**
    - **Concise:** Keep answers short. Use multiple small paragraphs.
    - **Highlighting:** Use **bold text** for key astrological terms.
    - **Grounded in Texts:** Your answer MUST synthesize the user's personal chart data with the provided ancient verses. You can explicitly reference the source text if it adds authority.

    **Your Task:** Analyze the provided User's Chart, the relevant Ancient Verses, and the Conversation History. Provide a concise, structured, and personalized answer to the LATEST User Question.

    ---
    **1. RELEVANT ANCIENT VERSES (Your Knowledge Base):**
    {retrieved_context}
    ---
    **2. USER'S BIRTH CHART DATA (Personalization):**
    {kundli_summary}
    ---
    **3. PREVIOUS CONVERSATION (Context):**
    {chat_history}
    ---
    **LATEST USER QUESTION:** "{question}"

    **PANDIT 2.0'S SYNTHESIZED RESPONSE:**
    """
    try:
        response_stream = llm_model.generate_content(prompt, stream=True)
        for chunk in response_stream:
            yield chunk.text
    except Exception as e:
        yield f"Error communicating with the AI Brain: {e}"

# --- --------------------------------------- ---
# --- 🎨 4. THE STREAMLIT USER INTERFACE 🎨 ---
# --- --------------------------------------- ---
st.set_page_config(page_title="Pandit 2.0 Demo", layout="centered")
st.title("✨ Pandit 2.0 - AI Astrologer")

# Initialize session state variables
if "kundli_data" not in st.session_state: st.session_state.kundli_data = None
if "messages" not in st.session_state: st.session_state.messages = []

# Sidebar for user input
with st.sidebar:
    st.header("Enter Birth Details")
    day = st.number_input("Day", 1, 31, 15)
    month = st.number_input("Month", 1, 12, 5)
    year = st.number_input("Year", 1990, 2023, 1995)
    hour = st.number_input("Hour (24h)", 0, 23, 14)
    minute = st.number_input("Minute", 0, 59, 30)
    tzone = st.number_input("Timezone", value=5.5, format="%.1f")
    city_name = st.text_input("City of Birth", "New Delhi, India")

    if st.button("Generate & Analyze Kundli", type="primary"):
        lat, lon = get_coordinates(gmaps_client, city_name)
        if lat and lon:
            with st.spinner("Calculating your cosmic blueprint..."):
                st.session_state.kundli_data = get_kundli_data(day, month, year, hour, minute, lat, lon, tzone)
                st.session_state.messages = []
                if st.session_state.kundli_data:
                    initial_greeting = "Namaste. Your charts have been generated and I have analyzed the core of your horoscope. I am now ready to answer your questions. How may I assist you?"
                    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
                    st.rerun()
        else:
            st.error(f"Could not find coordinates for '{city_name}'.")

# Main chat and chart area
if not st.session_state.kundli_data:
    st.info("👋 Welcome! Please provide your birth details in the sidebar to begin your personalized consultation.")
else:
    with st.expander("View Your Astrological Charts", expanded=False):
        chart_data = st.session_state.kundli_data.get('charts', {})
        if not chart_data and 'd1_chart_svg' in st.session_state.kundli_data: chart_data = st.session_state.kundli_data
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Lagna (D1)"); st.image(chart_data.get('d1_svg', 'https://via.placeholder.com/300x300.png?text=Chart+Not+Available'), use_column_width=True)
            st.subheader("Navamsa (D9)"); st.image(chart_data.get('d9_svg', 'https://via.placeholder.com/300x300.png?text=Chart+Not+Available'), use_column_width=True)
        with col2:
            st.subheader("Moon Chart"); st.image(chart_data.get('moon_chart_svg', 'https://via.placeholder.com/300x300.png?text=Chart+Not+Available'), use_column_width=True)
            st.subheader("Chalit Chart"); st.image(chart_data.get('chalit_chart_svg', 'https://via.placeholder.com/300x300.png?text=Chart+Not+Available'), use_column_width=True)
    
    st.divider()

    # Chat UI Logic
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Ask about your destiny..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting ancient texts and your stars..."):
                time.sleep(2) # Artificial "thinking" delay
                
                query_vector = embedding_model.encode(prompt).tolist()
                search_results = qdrant_client.search(
                    collection_name=COLLECTION_NAME, query_vector=query_vector, limit=3
                )
                retrieved_context = "\n\n---\n\n".join([res.payload['text_chunk'] for res in search_results])
                
                main_kundli_data = st.session_state.kundli_data.get('kundli_data', st.session_state.kundli_data)
                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                
                response_generator = get_llm_analysis(main_kundli_data, prompt, history_str, retrieved_context)
                
                full_response = st.write_stream(response_generator)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
