import os
import pandas as pd
from groq import Groq


from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)


#client = Groq()

def load_and_preprocess_mock_data():
    df = pd.read_csv('events.csv')
    df = df.sort_values(by='timestamp')
    return df
def get_user_history_string(df, user_id):
    """Filters the dataframe for a specific user and maps out their historical timeline."""
    user_actions = df[df['visitorid'] == user_id]
    
    if user_actions.empty:
        return "No history found for this user."
    
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
# --- EXECUTION PIPELINE ---
if __name__ == "__main__":
    print("--- Loading Retail Rocket Processed Events Data ---")
    events_df = load_and_preprocess_mock_data()
    
    # Let's extract the history of our target buyer (ID: 102019)
    target_user = 102019
    user_profile_text = get_user_history_string(events_df, target_user)
    
    print(f"\nConstructed Profile Text for User {target_user}:")
    print(user_profile_text)
    
    # Simulating standard candidate generation candidates (e.g., from an item popularity or BM25 retrieval step)
    candidates = [49521, 150318, 88888, 11111, 22222]
    print(f"\nCandidate Pool Size: {len(candidates)} items.")
    
    print("\n--- Generating LLM Recommendations via Groq API ---")
    recommendation_output = recommend_next_items(user_profile_text, candidates)
    
    print("\nEngine JSON Output Result:")
    print(recommendation_output)
