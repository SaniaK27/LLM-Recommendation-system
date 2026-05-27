import os
import pandas as pd
from groq import Groq
import streamlit as st
from dotenv import load_dotenv
import json

# 1. Page Configuration
st.set_page_config(page_title="LLM Recommender Engine", page_icon="🛍️", layout="centered")

st.title(" AI-Powered Retail Recommendation Engine")
st.caption("Final Year Project Demo | Inference: Groq API | Model: Llama 3.1")

# Load Environment Variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error(" Groq API Key missing! Please add GROQ_API_KEY to your Streamlit Secrets.")
    st.stop()

client = Groq(api_key=groq_api_key)

# --- 📦 PRODUCT CATALOG MATRIX ---
PRODUCT_CATALOG = {
    355908: {"name": "Sony WH-1000XM5 Wireless Headphones", "category": "Electronics", "price": "$399.00"},
    25656:  {"name": "Anker USB-C Fast Charging Cable (6ft)", "category": "Electronics Accessories", "price": "$19.99"},
    455502: {"name": "Apple MacBook Air M3 (13-inch)", "category": "Computers", "price": "$1,099.00"},
    118252: {"name": "Logitech MX Master 3S Ergonomic Mouse", "category": "Computer Accessories", "price": "$99.99"},
    
    # Candidate Pool Items
    49521:  {"name": "Apple AirPods Pro (2nd Generation)", "category": "Electronics", "price": "$249.00"},
    150318: {"name": "Satechi Aluminum Multi-Port USB-C Hub", "category": "Computer Accessories", "price": "$79.99"},
    88888:  {"name": "Dell UltraSharp 27-inch 4K Monitor", "category": "Computers & Displays", "price": "$459.99"},
    11111:  {"name": "Hydro Flask 32 oz Wide Mouth Water Bottle", "category": "Fitness & Outdoor", "price": "$44.95"},
    22222:  {"name": "Spigen Liquid Air iPhone 15 Pro Case", "category": "Mobile Accessories", "price": "$15.99"}
}

# --- 👥 USER PERSONA METADATA MAPPING ---
USER_PERSONA_MAP = {
    257597: {
        "name": "Rohan Malhotra",
        "role": "University Student (Tech & Gaming enthusiast)",
        "bio": "Daily commuter who studies on the go and spends evenings working on programming assignments and casual gaming."
    },
    999123: {
        "name": "Meera Nair",
        "role": "Remote Job Professional (Corporate Software Engineer)",
        "bio": "Working fully from her home office setup. Focuses heavily on desk ergonomics, productivity tools, and hardware efficiency."
    }
}

def get_product_details(item_id):
    return PRODUCT_CATALOG.get(int(item_id), {"name": f"Premium Retail Item #{item_id}", "category": "General Merchandise", "price": "$49.99"})

# --- CORE LOGIC ---
@st.cache_data
def load_and_preprocess_mock_data():
    df = pd.read_csv('events.csv')
    df = df.sort_values(by='timestamp')
    # Filter dataframe to track only our active demo personas
    full_user_list = list(df['visitorid'].unique())
    df_filtered = df[df['visitorid'].isin(USER_PERSONA_MAP.keys())]
    return df_filtered, len(df), len(full_user_list)

def get_user_history_string(df, user_id):
    user_actions = df[df['visitorid'] == user_id]
    if user_actions.empty:
        return None
    
    history_events = []
    for _, row in user_actions.iterrows():
        prod = get_product_details(row['itemid'])
        action_desc = f"- {row['event'].upper()}: {prod['name']}"
        history_events.append(action_desc)
        
    return "\n".join(history_events)

def recommend_next_items(user_persona, user_history, candidate_items, candidate_context_str):
    system_prompt = (
        "You are an expert e-commerce recommendation engine for a retail platform. "
        "Your task is to analyze a user's professional persona profile and past interaction behavior "
        "to determine which candidate items they are most likely to interact with next. "
        "Return your answer strictly in a valid JSON format with a key 'recommendations' mapping to a list of integer object IDs."
    )
    
    user_prompt = f"""
    User Professional Profile:
    - Name: {user_persona['name']}
    - Occupation/Role: {user_persona['role']}
    - Core Habits: {user_persona['bio']}

    User Interaction History (Oldest to Newest):
    {user_history}

    Available Candidate Items for Recommendation:
    {candidate_context_str}

    Task:
    Analyze both the structural sequence of past items AND how contextually appropriate the candidates 
    are for a person working in this user's specific career/lifestyle role. Pick the top 2 best candidate IDs.
    
    Expected JSON Output format:
    {{
        "explanation": "Provide a comprehensive reasoning matrix explaining why these specific items match both their historical sequence and professional career profile.",
        "recommendations": [item_id_1, item_id_2]
    }}
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, 
            response_format={"type": "json_object"} 
        )
        return completion.choices[0].message.content, system_prompt, user_prompt
    except Exception as e:
        return f"An error occurred with the API call: {e}", "", ""

# --- STREAMLIT UI LAYOUT & EXECUTION ---
try:
    events_df, total_raw_rows, total_unique_users = load_and_preprocess_mock_data()
except FileNotFoundError:
    st.error("📂 `events.csv` missing from your main repository directory.")
    st.stop()

# UPGRADE 1: Dataset Overview Metrics Card Header
col1, col2, col3 = st.columns(3)
col1.metric("Total Click Events", f"{total_raw_rows:,}")
col2.metric("Tracked System Profiles", f"{total_unique_users:,}")
col3.metric("Candidate Pool Size", "5 Items")
st.write("---")

# Sidebar Configuration Layout
st.sidebar.header(" Simulation Control Panel")

# UPGRADE 3 (Part 1): Set up the Presentation Toggle Mode
app_mode = st.sidebar.radio("View Dashboard Mode As:", [" Executive Dashboard", " Research / Debug Mode"])

st.sidebar.write("---")

# User Selection Dropdown Setup
dropdown_options = {v_id: f" {meta['name']} ({meta['role']})" for v_id, meta in USER_PERSONA_MAP.items()}
selected_user_id = st.sidebar.selectbox(
    "Select a User Profile to Test:", 
    options=list(dropdown_options.keys()), 
    format_func=lambda x: dropdown_options[x]
)

current_persona = USER_PERSONA_MAP[selected_user_id]

# Predefined Candidate Item Pool configuration
candidates = [49521, 150318, 88888, 11111, 22222]
candidates_context = ""
for c in candidates:
    det = get_product_details(c)
    candidates_context += f"- ID {c}: {det['name']} (Category: {det['category']}, Price: {det['price']})\n"

st.sidebar.markdown("### Available Candidate Inventory")
for c in candidates:
    det = get_product_details(c)
    st.sidebar.caption(f"**ID {c}**: {det['name']} ({det['price']})")

# Main Stage Display Elements
st.markdown(f"### Target Persona Focus: **{current_persona['name']}**")
st.markdown(f" **Professional Career Designation:** `{current_persona['role']}`")
st.caption(f" **User Profile Persona Summary:** {current_persona['bio']}")

user_profile_text = get_user_history_string(events_df, selected_user_id)

if user_profile_text is None:
    st.warning("No data history context found.")
