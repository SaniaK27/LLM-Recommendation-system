import os
import pandas as pd
from groq import Groq
import streamlit as st
from dotenv import load_dotenv
import json

# 1. Page Configuration & Theme setup
st.set_page_config(page_title="LLM Recommender Engine", page_icon="🛍️", layout="centered")

st.title("🛍️ LLM-Based Sequential Recommendation Engine")
st.caption("Final Year Project Demo | Dataset: Retail Rocket | Inference: Groq API")
st.write("---")

# 2. Retrieve the Groq API key safely from environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("🔑 Groq API Key missing! Please add GROQ_API_KEY to your Streamlit Secrets dashboard.")
    st.stop()

# Initialize Groq Client
client = Groq(api_key=groq_api_key)

# 3. Core Logic Functions
@st.cache_data # Caches data so it doesn't reload from disk on every button click
def load_and_preprocess_mock_data():
    df = pd.read_csv('events.csv')
    df = df.sort_values(by='timestamp')
    return df

def get_user_history_string(df, user_id):
    """Filters the dataframe for a specific user and maps out their historical timeline."""
    user_actions = df[df['visitorid'] == user_id]
    
    if user_actions.empty:
        return None
    
    history_events = []
    for _, row in user_actions.iterrows():
        action_desc = f"- {row['event'].upper()} item ID: {row['itemid']}"
        history_events.append(action_desc)
        
    return "\n".join(history_events)

def recommend_next_items(user_history, candidate_items):
    """Calls Groq API to perform LLM-based zero-shot sequential recommendation."""
    system_prompt = (
        "You are an expert e-commerce recommendation engine for a retail platform. "
        "Your task is to analyze a user's past interaction behavior (views, additions to cart, and transactions) "
        "and determine which items from a list of available candidates they are most likely to interact with next. "
        "Return your answer strictly in a valid JSON format with a key 'recommendations' mapping to a list of object IDs."
    )
    
    user_prompt = f"""
    User Interaction History (Oldest to Newest):
    {user_history}

    Available Candidate Item IDs for Recommendation:
    {candidate_items}

    Task:
    Analyze patterns in the item IDs they interacted with. Pick the top 2 best candidate IDs 
    that logically match the user's sequential consumption behavior. 
    
    Expected JSON Output format:
    {{
        "explanation": "Brief reasoning for why these items fit the past sequence",
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

# --- STREAMLIT UI PIPELINE ---

# Load Data
try:
    events_df = load_and_preprocess_mock_data()
except FileNotFoundError:
    st.error("📂 `events.csv` not found in the root directory! Please upload it to your repository.")
    st.stop()

# Sidebar Setup for Interactive Presentation
st.sidebar.header("🕹️ Control Panel")
# Automatically pull unique visitor IDs from your events.csv file
unique_users = events_df['visitorid'].unique()
selected_user = st.sidebar.selectbox("Select a Target Buyer ID:", unique_users)

# Define candidates
candidates = [49521, 150318, 88888, 11111, 22222]
st.sidebar.write(f"**Candidate Pool:** {candidates}")

# Main Layout Presentation
st.subheader(f"👤 Customer Profile: Visitor ID {selected_user}")

user_profile_text = get_user_history_string(events_df, selected_user)

if user_profile_text is None:
    st.warning(f"No history found in mock dataset for user {selected_user}. Please select another ID from the sidebar.")
else:
    # Display the constructed narrative
    st.info("**Constructed Behavioral Narrative (Fed to LLM):**")
    st.code(user_profile_text, language="text")

    # The Action Button
    if st.button("🚀 Run LLM Recommender Engine"):
        with st.spinner("Analyzing sequences via Llama-3.1 on Groq..."):
            recommendation_output = recommend_next_items(user_profile_text, candidates)
            
            st.success("🤖 Recommendation Framework Execution Successful!")
            
            # Display results in two clean tabs
            tab1, tab2 = st.tabs(["📊 Structured Output", "⚙️ Raw JSON JSON Response"])
            
            with tab1:
                try:
                    # Attempt to parse the JSON string response
                    parsed_json = json.loads(recommendation_output)
                    
                    st.markdown("### 🎯 Top Recommended Items")
                    cols = st.columns(len(parsed_json['recommendations']))
                    for idx, item in enumerate(parsed_json['recommendations']):
                        cols[idx].metric(label=f"Recommendation #{idx+1}", value=f"ID: {item}")
                    
                    st.markdown("### 🧠 LLM Reasoning Matrix")
                    st.write(parsed_json['explanation'])
                    
                except Exception:
                    st.write("Could not parse output cleanly into cards. See raw JSON tab.")
                    st.write(recommendation_output)
            
            with tab2:
                st.json(recommendation_output)
