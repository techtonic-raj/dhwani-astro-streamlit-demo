# import streamlit as st
# import httpx
# import json
# import base64
# import google.generativeai as genai
# import googlemaps
# from qdrant_client import QdrantClient
# from sentence_transformers import SentenceTransformer
# import time

# # --- ----------------------------------------------------------------- ---
# # --- ⚙️ 1. SECURE CONFIGURATION (Uses Streamlit Secrets) ⚙️ ---
# # --- ----------------------------------------------------------------- ---

# # This section tells the app to look for secrets provided by the Streamlit hosting environment.
# # You will paste your actual keys into the Streamlit deployment settings, NOT here.

# try:
#     # Astrology API Credentials
#     ASTRO_API_URL = st.secrets["ASTRO_API_URL"]
#     ASTRO_API_USER_ID = st.secrets["ASTRO_API_USER_ID"]
#     ASTRO_API_KEY = st.secrets["ASTRO_API_KEY"]

#     # Google Services API Keys
#     GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
#     GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]

#     # Qdrant Cloud Credentials
#     QDRANT_URL = st.secrets["QDRANT_URL"]
#     QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
#     COLLECTION_NAME = "pandit_books"
    
#     # Configure the Gemini client
#     genai.configure(api_key=GEMINI_API_KEY)

# except KeyError as e:
#     # If a secret is missing, stop the app and show a helpful error
#     st.error(f"🚨 Missing Secret! Please add '{e.args[0]}' to your Streamlit Cloud secrets.")
#     st.stop()


# # --- -------------------------------------------------- ---
# # --- 🧠 2. INITIALIZE AI MODELS & SERVICES (Cached) 🧠 ---
# # --- -------------------------------------------------- ---

# @st.cache_resource
# def get_services():
#     """Initializes all the necessary clients and models safely."""
#     llm_model = genai.GenerativeModel('gemini-1.5-pro-latest')
#     gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
#     qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
#     embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
#     return llm_model, gmaps_client, qdrant_client, embedding_model

# llm_model, gmaps_client, qdrant_client, embedding_model = get_services()

# # --- -------------------------------------------- ---
# # --- 📞 3. API & CORE LOGIC FUNCTIONS 📞 ---
# # --- -------------------------------------------- ---

# @st.cache_data
# def get_coordinates(_gmaps_client, city_name: str):
#     # ... (This function remains the same) ...
#     try:
#         geocode_result = _gmaps_client.geocode(city_name)
#         if geocode_result: return geocode_result[0]['geometry']['location']['lat'], geocode_result[0]['geometry']['location']['lng']
#     except Exception as e:
#         st.error(f"Geocoding Error: {e}")
#     return None, None

# def get_kundli_data(day, month, year, hour, minute, lat, lon, tzone):
#     """
#     CORRECTED: This function now uses the GET method and sends data as params.
#     """
#     try:
#         payload = {"day": day, "month": month, "year": year, "hour": hour, "min": minute, "lat": lat, "lon": lon, "tzone": tzone, "charts_format": "svg"}
#         auth_string = f"{ASTRO_API_USER_ID}:{ASTRO_API_KEY}"
#         b64_auth = base64.b64encode(auth_string.encode()).decode()
#         headers = {"Authorization": f"Basic {b64_auth}"} # Content-Type is not needed for GET
        
#         with httpx.Client(timeout=30.0) as client:
#             # --- THE FIX: Changed .post to .get and json= to params= ---
#             response = client.get(ASTRO_API_URL, params=payload, headers=headers)
#             response.raise_for_status()
#         return response.json()
#     except Exception as e:
#         st.error(f"Failed to fetch Kundli data: {e}")
#         return None

# def get_llm_analysis(kundli_json: dict, question: str, chat_history: str, retrieved_context: str):
#     """Sends the complete context to the Gemini model and streams the response."""
#     kundli_summary = json.dumps(kundli_json, indent=2)
#     prompt = f"""You are Pandit 2.0, a world-class Vedic Astrologer.

#     **Your Persona:**
#     - **Tone:** Wise, confident, empathetic, and direct.
#     - **Greetings:** Always begin every response with a respectful greeting.

#     **Your Answer Structure (CRUCIAL):**
#     - **Concise:** Keep answers short. Use multiple small paragraphs.
#     - **Highlighting:** Use **bold text** for key astrological terms.
#     - **Grounded in Texts:** Your answer MUST synthesize the user's personal chart data with the provided ancient verses. You can explicitly reference the source text if it adds authority.

#     **Your Task:** Analyze the provided User's Chart, the relevant Ancient Verses, and the Conversation History. Provide a concise, structured, and personalized answer to the LATEST User Question.

#     ---
#     **1. RELEVANT ANCIENT VERSES (Your Knowledge Base):**
#     {retrieved_context}
#     ---
#     **2. USER'S BIRTH CHART DATA (Personalization):**
#     {kundli_summary}
#     ---
#     **3. PREVIOUS CONVERSATION (Context):**
#     {chat_history}
#     ---
#     **LATEST USER QUESTION:** "{question}"

#     **PANDIT 2.0'S SYNTHESIZED RESPONSE:**
#     """
#     try:
#         response_stream = llm_model.generate_content(prompt, stream=True)
#         for chunk in response_stream:
#             yield chunk.text
#     except Exception as e:
#         yield f"Error communicating with the AI Brain: {e}"

# # --- --------------------------------------- ---
# # --- 🎨 4. THE STREAMLIT USER INTERFACE 🎨 ---
# # --- --------------------------------------- ---
# st.set_page_config(page_title="Pandit 2.0 Demo", layout="centered")
# st.title("✨ Pandit 2.0 - AI Astrologer")

# # Initialize session state variables
# if "kundli_data" not in st.session_state: st.session_state.kundli_data = None
# if "messages" not in st.session_state: st.session_state.messages = []

# # Sidebar for user input
# with st.sidebar:
#     st.header("Enter Birth Details")
#     day = st.number_input("Day", 1, 31, 15)
#     month = st.number_input("Month", 1, 12, 5)
#     year = st.number_input("Year", 1990, 2023, 1995)
#     hour = st.number_input("Hour (24h)", 0, 23, 14)
#     minute = st.number_input("Minute", 0, 59, 30)
#     tzone = st.number_input("Timezone", value=5.5, format="%.1f")
#     city_name = st.text_input("City of Birth", "New Delhi, India")

#     if st.button("Generate & Analyze Kundli", type="primary"):
#         lat, lon = get_coordinates(gmaps_client, city_name)
#         if lat and lon:
#             with st.spinner("Calculating your cosmic blueprint..."):
#                 st.session_state.kundli_data = get_kundli_data(day, month, year, hour, minute, lat, lon, tzone)
#                 st.session_state.messages = []
#                 if st.session_state.kundli_data:
#                     initial_greeting = "Namaste. Your charts have been generated and I have analyzed the core of your horoscope. I am now ready to answer your questions. How may I assist you?"
#                     st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
#                     st.rerun()
#         else:
#             st.error(f"Could not find coordinates for '{city_name}'.")

# # Main chat and chart area
# if not st.session_state.kundli_data:
#     st.info("👋 Welcome! Please provide your birth details in the sidebar to begin your personalized consultation.")
# else:
#     with st.expander("View Your Astrological Charts", expanded=False):
#         chart_data = st.session_state.kundli_data.get('charts', {})
#         if not chart_data and 'd1_chart_svg' in st.session_state.kundli_data: chart_data = st.session_state.kundli_data
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.subheader("Lagna (D1)"); st.image(chart_data.get('d1_svg', 'https://via.placeholder.com/300x300.png?text=Chart+Not+Available'), use_column_width=True)
#             st.subheader("Navamsa (D9)"); st.image(chart_data.get('d9_svg', 'https://via.placeholder.com/300x300.png?text=Chart+Not+Available'), use_column_width=True)
#         with col2:
#             st.subheader("Moon Chart"); st.image(chart_data.get('moon_chart_svg', 'https://via.placeholder.com/300x300.png?text=Chart+Not+Available'), use_column_width=True)
#             st.subheader("Chalit Chart"); st.image(chart_data.get('chalit_chart_svg', 'https://via.placeholder.com/300x300.png?text=Chart+Not+Available'), use_column_width=True)
    
#     st.divider()

#     # Chat UI Logic
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"], unsafe_allow_html=True)

#     if prompt := st.chat_input("Ask about your destiny..."):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             with st.spinner("Consulting ancient texts and your stars..."):
#                 time.sleep(2) # Artificial "thinking" delay
                
#                 query_vector = embedding_model.encode(prompt).tolist()
#                 search_results = qdrant_client.search(
#                     collection_name=COLLECTION_NAME, query_vector=query_vector, limit=3
#                 )
#                 retrieved_context = "\n\n---\n\n".join([res.payload['text_chunk'] for res in search_results])
                
#                 main_kundli_data = st.session_state.kundli_data.get('kundli_data', st.session_state.kundli_data)
#                 history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                
#                 response_generator = get_llm_analysis(main_kundli_data, prompt, history_str, retrieved_context)
                
#                 full_response = st.write_stream(response_generator)
        
#         st.session_state.messages.append({"role": "assistant", "content": full_response})




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
    """Initializes all the necessary clients and a single, versatile AI model."""
    # --- MODIFIED: Using the user-specified preview model for all AI tasks. ---
    llm_model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
    
    gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    return llm_model, gmaps_client, qdrant_client, embedding_model

# Unpack the single model and other services
llm_model, gmaps_client, qdrant_client, embedding_model = get_services()


# --- -------------------------------------------- ---
# --- 📞 3. API & CORE LOGIC FUNCTIONS 📞 ---
# --- -------------------------------------------- ---

# --- TASK 1: THE GUARDRAIL / FILTER ---
def filter_user_question(question: str):
    """Uses the LLM to classify the user's question."""
    prompt = f"""
    Analyze the user's question and classify it into one of two categories: "ASTROLOGY" or "META".
    - "ASTROLOGY" questions are about horoscopes, planets, life events (marriage, career), etc.
    - "META" questions are about you (the AI), how you are built, or other off-topic subjects.

    User question: "{question}"
    Category:
    """
    try:
        response = llm_model.generate_content(prompt)
        category = response.text.strip().upper()
        if "ASTROLOGY" in category:
            return "ASTROLOGY"
        return "META"
    except Exception as e:
        st.error(f"Error in question filter: {e}")
        return "ASTROLOGY" # Fail safe

# --- TASK 2: THE DRAFTER ---
def get_initial_llm_draft(kundli_json: dict, question: str, chat_history: str):
    """Generates a first-pass analysis based only on the user's chart."""
    kundli_summary = json.dumps(kundli_json, indent=2)
    prompt = f"""You are a junior Vedic Astrologer. Provide a direct, preliminary analysis based *only* on the provided birth chart and question. Be factual and concise.

    *USER'S BIRTH CHART DATA:*
    {kundli_summary}
    ---
    *CONVERSATION HISTORY:*
    {chat_history}
    ---
    *LATEST USER QUESTION:* "{question}"

    *JUNIOR PANDIT'S PRELIMINARY DRAFT:*
    """
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error during initial draft generation: {e}"

# --- TASK 3: THE REVIEWER ---
def get_final_reviewed_answer(initial_draft: str, retrieved_context: str, kundli_json: dict, question: str, chat_history: str):
    """The same model, now acting as a senior reviewer, synthesizes all info for a final answer."""
    kundli_summary = json.dumps(kundli_json, indent=2)
    prompt = f"""You are Pandit 2.0, a world-class Senior Vedic Astrologer, reviewing a junior's draft.

    *Your Task:* Produce a final, wise answer by synthesizing three sources:
    1.  *The Junior Pandit's Draft.*
    2.  *Relevant Ancient Verses.*
    3.  *The User's Full Chart & History.*

    *Your Persona & Structure:*
    - *Tone:* Wise, confident, empathetic, direct.
    - *Greeting:* Always begin with a respectful greeting like "Namaste."
    - *Synthesis:* Your answer MUST integrate insights from the ancient verses with the specifics of the user's chart. Use the junior's draft for ideas, but form your own superior conclusion.
    - *Formatting:* Use *bold text* for key astrological terms and short, clear paragraphs.

    ---
    *1. RELEVANT ANCIENT VERSES (Your Knowledge Base):*
    {retrieved_context}
    ---
    *2. JUNIOR PANDIT'S DRAFT (For your review):*
    {initial_draft}
    ---
    *3. USER'S BIRTH CHART DATA (Personalization):*
    {kundli_summary}
    ---
    *4. PREVIOUS CONVERSATION (Context):*
    {chat_history}
    ---
    *LATEST USER QUESTION:* "{question}"

    *PANDIT 2.0'S FINAL, SYNTHESIZED RESPONSE:*
    """
    try:
        response_stream = llm_model.generate_content(prompt, stream=True)
        for chunk in response_stream:
            yield chunk.text
    except Exception as e:
        yield f"Error communicating with the Senior AI Brain: {e}"

# (get_coordinates and get_kundli_and_charts functions remain the same)
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
                    all_data['charts'][key] = data.get('svg')
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

# (Sidebar and chart display code remains unchanged)
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
                st.session_state.kundli_data = get_kundli_and_charts(day, month, year, hour, minute, lat, lon, tzone)
                st.session_state.messages = []
                if st.session_state.kundli_data:
                    initial_greeting = "Namaste. Your charts have been generated. I have analyzed the core of your horoscope and am ready to answer your questions. How may I assist you?"
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
            st.image(chart_data.get('d1_svg', ''))
        with col2:
            st.subheader("Navamsa (D9)")
            st.image(chart_data.get('d9_svg', ''))
        with col1:
            st.subheader("Moon Chart")
            st.image(chart_data.get('moon_svg', ''))
        with col2:
            st.subheader("Chalit Chart")
            st.image(chart_data.get('chalit_svg', ''))
    
    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Ask about your destiny..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            # --- STEP 1: FILTER THE QUESTION ---
            category = filter_user_question(prompt)
            
            if category == "META":
                canned_response = "Namaste. I am Pandit 2.0, an AI assistant for Vedic Astrology. My purpose is to provide guidance based on your birth chart. Please ask questions related to your horoscope."
                st.write(canned_response)
                st.session_state.messages.append({"role": "assistant", "content": canned_response})
            
            else: # --- The question is about ASTROLOGY ---
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
