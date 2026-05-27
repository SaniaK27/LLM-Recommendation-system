import os
import pandas as pd
from groq import Groq
import streamlit as st
from dotenv import load_dotenv
import json

# Page Configuration
st.set_page_config(page_title="LLM Recommender Engine", page_icon="🛍️", layout="centered")

st.title(" AI-Powered Retail Recommendation Engine")
st.caption("Final Year Project Demo | Inference: Groq API | Model: Llama 3.1")
st.write("---")

# Load Environment Variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("🔑 Groq API Key missing! Please add GROQ_API_KEY to your Streamlit Secrets.")
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
# This maps the raw visitor IDs from events.csv to realistic human professional profiles
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
    # Filter the dataframe so it ONLY contains the users we have mapped personas for
    df = df[df['visitorid'].isin(USER_PERSONA_MAP.keys())]
    return df

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

def recommend_next_items(user_persona, user_history, candidate_items):
    candidate_context = ""
    for c_id in candidate_items:
        p_details = get_product_details(c_id)
        candidate_context += f"- ID {c_id}: {p_details['name']} (Category: {p_details['category']}, Price: {p_details['price']})\n"

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
    {candidate_context}

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
        return completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the API call: {e}"

# --- STREAMLIT UI LAYOUT ---
try:
    events_df = load_and_preprocess_mock_data()
except FileNotFoundError:
    st.error("📂 `events.csv` missing.")
    st.stop()

# Sidebar UI
st.sidebar.header(" Simulation Control Panel")

# Create clean dropdown text combining Name and Professional Title
dropdown_options = {v_id: f"👤 {meta['name']} ({meta['role']})" for v_id, meta in USER_PERSONA_MAP.items()}
selected_user_id = st.sidebar.selectbox(
    "Select a User Profile to Test:", 
    options=list(dropdown_options.keys()), 
    format_func=lambda x: dropdown_options[x]
)

# Fetch current persona metadata
current_persona = USER_PERSONA_MAP[selected_user_id]

# Candidate Display
candidates = [49521, 150318, 88888, 11111, 22222]
st.sidebar.markdown("###  Candidate Inventory Pool")
for c in candidates:
    det = get_product_details(c)
    st.sidebar.caption(f"**ID {c}**: {det['name']} ({det['price']})")

# Main Page UI Layout
st.markdown(f"### Target Persona: **{current_persona['name']}**")
st.markdown(f" **Professional Role:** `{current_persona['role']}`")
st.caption(f" **User Background Summary:** {current_persona['bio']}")
st.write("---")

user_profile_text = get_user_history_string(events_df, selected_user_id)

if user_profile_text is None:
    st.warning("No data found.")
else:
    st.info("**Extracted Behavioral Context (Passed to Llama-3.1):**")
    st.code(user_profile_text, language="text")

    if st.button(" Execute Persona-Aware LLM Recommendation"):
        with st.spinner("Analyzing cross-category behavioral matrices via Groq..."):
            recommendation_output = recommend_next_items(current_persona, user_profile_text, candidates)
            
            try:
                parsed_json = json.loads(recommendation_output)
                
                st.markdown("###  Top Personalized Recommendations")
                rec_ids = parsed_json['recommendations']
                cols = st.columns(len(rec_ids))
                
                for idx, item_id in enumerate(rec_ids):
                    prod_info = get_product_details(item_id)
                    with cols[idx]:
                        st.metric(label=f"Top Match #{idx+1}", value=prod_info['price'])
                        st.markdown(f"**{prod_info['name']}**")
                        st.caption(f"Category: {prod_info['category']} | ID: {item_id}")
                
                st.write("---")
                st.markdown("###  Neural Reasoning Matrix (Explainable AI)")
                st.success(parsed_json['explanation'])
                
            except Exception:
                st.error("Parsing failed. Raw dump:")
                st.write(recommendation_output)
