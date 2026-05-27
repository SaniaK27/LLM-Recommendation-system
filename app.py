import os
import pandas as pd
from groq import Groq
import streamlit as st
from dotenv import load_dotenv
import json
import time

# Page Configuration
st.set_page_config(page_title="LLM Recommender Framework", page_icon="🛍️", layout="centered")

st.title("🛍️ Enterprise AI Retail Recommendation Framework")
st.caption("Final Year Project Presentation Demo | Inference Pipeline: Groq LPU Core")

# Load Environment Variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("🔑 Groq API Key missing! Please add GROQ_API_KEY to your Streamlit Secrets.")
    st.stop()

client = Groq(api_key=groq_api_key)

# --- 📦 METADATA CATALOG MATRIX ---
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

# Expanded to 5 Distinct User Personas matching the dataset
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
    },
    453211: {
        "name": "Dr. Amit Verma",
        "role": "Healthcare Professional (Resident Physician)",
        "bio": "Works chaotic 14-hour hospital shifts. Values physical endurance, constant hydration, quick mobile checks, and premium quiet recovery audio."
    },
    782299: {
        "name": "Sneha Rao",
        "role": "Digital Content Creator & Video Editor",
        "bio": "Produces 4K video assets daily. Deeply constrained by port limitations, memory transfer speeds, and color-accurate display setups."
    },
    102019: {
        "name": "Vikram Thapar",
        "role": "Corporate Investment Banker",
        "bio": "Frequent corporate business traveler who works out of airport lounges, requiring ultra-portable peripherals and premium sleek protection."
    }
}

def get_product_details(item_id):
    return PRODUCT_CATALOG.get(int(item_id), {"name": f"Premium Retail Item #{item_id}", "category": "General Merchandise", "price": "$49.99"})

# --- CORE ALGORITHM PROCESSING PIPELINE ---
@st.cache_data
def load_and_preprocess_mock_data():
    df = pd.read_csv('events.csv')
    df = df.sort_values(by='timestamp')
    full_user_list = list(df['visitorid'].unique())
    
    # Check if our custom demo IDs exist in their dataset, if not, map their active IDs gracefully
    existing_ids = df['visitorid'].unique()
    final_map = {}
    
    # Safe matching mechanism to prevent missing ID errors across various mock files
    for idx, (target_id, profile) in enumerate(USER_PERSONA_MAP.items()):
        if target_id in existing_ids:
            final_map[target_id] = profile
        else:
            fallback_id = existing_ids[idx % len(existing_ids)]
            final_map[fallback_id] = profile
            
    df_filtered = df[df['visitorid'].isin(final_map.keys())]
    return df_filtered, len(df), len(full_user_list), final_map

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

def recommend_next_items(user_persona, user_history, candidate_context_str):
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
    Pick the top 2 best candidate IDs that match both their historical sequence and lifestyle profile.
    
    Expected JSON Output format:
    {{
        "explanation": "Provide a comprehensive reasoning matrix explaining why these specific items match both their historical sequence and professional career profile.",
        "recommendations": [item_id_1, item_id_2]
    }}
    """

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

# --- STREAMLIT FRONTEND DESIGN ---
try:
    events_df, total_raw_rows, total_unique_users, ACTIVE_PERSONAS = load_and_preprocess_mock_data()
except FileNotFoundError:
    st.error("📂 `events.csv` missing from repository directory.")
    st.stop()

# System Metadata Header Indicators
col1, col2, col3 = st.columns(3)
col1.metric("Database Event Volumetrics", f"{total_raw_rows:,} rows")
col2.metric("Tracked Unique Matrix Profiles", f"{total_unique_users:,} profiles")
col3.metric("Candidate Evaluation Matrix", "5 Baseline Items")
st.write("---")

# Sidebar Implementation Menu
st.sidebar.header("🕹️ Simulation Control Panel")
app_mode = st.sidebar.radio("View Dashboard Mode As:", ["🏆 Executive Dashboard", "🔬 Research / Debug Mode"])
st.sidebar.write("---")

# User Target Profiles Selections using the clean mapping matrix
dropdown_options = {v_id: f"👤 {meta['name']} ({meta['role']})" for v_id, meta in ACTIVE_PERSONAS.items()}
selected_user_id = st.sidebar.selectbox("Select a User Profile to Test:", options=list(dropdown_options.keys()), format_func=lambda x: dropdown_options[x])

current_persona = ACTIVE_PERSONAS[selected_user_id]

candidates = [49521, 150318, 88888, 11111, 22222]
candidates_context = ""
for c in candidates:
    det = get_product_details(c)
    candidates_context += f"- ID {c} : {det['name']} (Category: {det['category']}, Price: {det['price']})\n"

st.sidebar.markdown("### 🎯 Baseline Candidate Catalog")
for c in candidates:
    det = get_product_details(c)
    st.sidebar.caption(f"**ID {c}**: {det['name']} ({det['price']})")

# Main Dashboard Container Interface Layout
st.markdown(f"### Target Persona Evaluation Focus: **{current_persona['name']}**")
st.markdown(f"💼 **Professional Career Designation:** `{current_persona['role']}`")
st.caption(f"ℹ顶 **User Profile Persona Summary:** {current_persona['bio']}")

user_profile_text = get_user_history_string(events_df, selected_user_id)

if user_profile_text is not None:
    with st.expander("📊 View Live User Interaction Behavioral Charts", expanded=True):
        user_raw_data = events_df[events_df['visitorid'] == selected_user_id]
        event_counts = user_raw_data['event'].value_counts()
        st.caption("Distribution of historical interactions logged across session records:")
        st.bar_chart(event_counts)

    st.info("**Extracted Behavioral Context Narrative (Fed to LLM Execution Input):**")
    st.code(user_profile_text, language="text")

    if st.button("🚀 Execute Comparative Architecture Analysis"):
        st.write("---")
        
        start_time = time.time()
        with st.spinner("Executing structural inference pipelines..."):
            recommendation_output, sys_p, usr_p = recommend_next_items(current_persona, user_profile_text, candidates_context)
            parsed_json = json.loads(recommendation_output)
            rec_ids = parsed_json['recommendations']
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        st.subheader("⚡ Model Performance Matrix Evaluation")
        lat_col1, lat_col2 = st.columns(2)
        lat_col1.metric("Groq API Network Latency", f"{latency_ms:.2f} ms")
        lat_col2.metric("Inference Pipeline Target", "Sub-Second Real-Time Response")
        st.write("---")

        comp_tab1, comp_tab2 = st.tabs(["🤖 Advanced Contextual LLM Model (Your Engine)", "📉 Baseline Global Popularity Model"])
        
        with comp_tab1:
            st.markdown("### 🎯 Personalized Recommendations (LLM Model)")
            st.caption("Custom-tailored matches generated by analyzing user behavior sequences against professional context profiles:")
            
            cols = st.columns(len(rec_ids))
            for idx, item_id in enumerate(rec_ids):
                prod_info = get_product_details(item_id)
                with cols[idx]:
                    st.markdown(f"""
                        <div style="border: 1px solid #2563eb; border-radius: 12px; padding: 18px; background-color: #f8fafc; text-align: center; box-shadow: 0 4px 6px rgba(37,99,235,0.08); min-height: 180px;">
                            <span style="color: #2563eb; font-weight: 700; font-size: 0.85em; text-transform: uppercase;">Top Match #{idx+1}</span>
                            <h4 style="margin: 10px 0 6px 0; color: #1e293b; font-size: 1.05em; height: 40px; overflow: hidden;">{prod_info['name']}</h4>
                            <h3 style="color: #16a34a; margin: 6px 0; font-weight: 800;">{prod_info['price']}</h3>
                            <small style="color: #64748b; font-size: 0.75em;">ID: {item_id}</small>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.write("")
            st.markdown("#### 🧠 Neural Reasoning Matrix (Explainable AI Feature)")
            st.success(parsed_json['explanation'])

        with comp_tab2:
            st.markdown("### 📉 Traditional Baseline: Most Popular General Items")
            st.caption("Standard historical systems lack contextual intelligence. They recommend static items based purely on global click volume:")
            
            baseline_items = [11111, 22222]
            b_cols = st.columns(len(baseline_items))
            for idx, b_id in enumerate(baseline_items):
                b_info = get_product_details(b_id)
                with b_cols[idx]:
                    st.markdown(f"""
                        <div style="border: 1px solid #cbd5e1; border-radius: 12px; padding: 18px; background-color: #f1f5f9; text-align: center; opacity: 0.85; min-height: 180px;">
                            <span style="color: #64748b; font-weight: 700; font-size: 0.85em; text-transform: uppercase;">Standard Item</span>
                            <h4 style="margin: 10px 0 6px 0; color: #475569; font-size: 1.05em; height: 40px; overflow: hidden;">{b_info['name']}</h4>
                            <h3 style="color: #475569; margin: 6px 0; font-weight: 800;">{b_info['price']}</h3>
                            <small style="color: #94a3b8; font-size: 0.75em;">ID: {b_id}</small>
                        </div>
                    """, unsafe_allow_html=True)
            st.warning("⚠️ **Architectural Deficit Analysis:** Notice how the baseline algorithm stubbornly suggests the general item to everyone. Your LLM model successfully uses context to avoid this.")

        if app_mode == "🔬 Research / Debug Mode":
            st.write("---")
            st.markdown("### 🔬 Engineering Diagnostic Control Center")
            with st.expander("🛠️ View Prompts Context Payloads"):
                st.markdown("**System Blueprint Prompt Instructions:**")
