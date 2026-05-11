import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd 
import streamlit as st
from core.simulation_engine import simulate_behavior, behavioral_interpretation
from core.simulation_engine import scenarios, difficulty_map
st.set_page_config(page_title = "HDT - Behavior Simulation", layout = "wide")
st.markdown(""" <style> body {background-color: #f5f7fa;} h1 {color: #1f2937;} h2, h3 {color: #374151;} .stButton > button {background-color: #2563eb; color: white; border-radius: 8px; height: 45px; font-size: 16px;} </style> """, unsafe_allow_html = True)
st.title("Human Digital Twin for Behavior Simulation Using AI")
st.markdown("### HDT System Interface")
st.divider()
st.subheader("Enter Situation Details")
user_text = st.text_area("Describe your situation:", height = 120)
selected_scenario = st.selectbox("Select Scenario:", scenarios)
selected_difficulty = st.selectbox("Select Difficulty Level:", difficulty_map[selected_scenario])
st.divider()
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    run_button = st.button("Run Simulation")
if run_button:
    if user_text.strip() == "":
        st.warning("Please enter a situation description.")
    else:
        result = simulate_behavior(user_text = user_text, scenario = selected_scenario, difficulty = selected_difficulty)
        interpretation_text = behavioral_interpretation(mapped_emotion = result["Mapped Emotion"], difficulty = selected_difficulty, decision = result["Predicted Decision"])
        st.divider()
        st.markdown("## Behavioral Simulation Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Decision", result["Predicted Decision"])
            st.metric("Detected Emotion", result["Detected Emotion"])
        with col2:
            st.metric("Mapped Emotion", result["Mapped Emotion"])
            confidence_percent = round(result["Confidence"] * 100, 1)
            st.metric("Confidence Level", f"{confidence_percent}%")
        st.divider()
        st.subheader("Emotion Confidence Visualization")
        emotion_df = pd.DataFrame({"Emotion": [result["Detected Emotion"]], "Confidence": [result["Confidence"]]})
        fig_emotion = px.bar(emotion_df, x = "Emotion", y = "Confidence", color = "Emotion", text = "Confidence", title = "Detected emotion confidence level")
        fig_emotion.update_traces(textposition = "outside")
        fig_emotion.update_layout(yaxis_range = [0, 1.1], height = 500, width = 500)
        col1, col2, col3 = st.columns([2, 3, 2])
        with col2:
            st.plotly_chart(fig_emotion)
        st.divider()
        st.subheader("Predicted Decision Tendency")
        decision = result["Predicted Decision"]
        decision_df = pd.DataFrame({"Decision": [decision], "Confidence": [result["Confidence"]], "Label": [f"{confidence_percent}%"]})
        fig_decision = px.bar(decision_df, x = "Decision", y = "Confidence", color = "Decision", text = "Confidence", title = "Predicted Behavioral Decision")
        fig_decision.update_traces(textposition = "outside")
        fig_decision.update_layout(yaxis_range = [0, 1.1], height = 500, width = 500)
        col1, col2, col3 = st.columns([2, 3, 2])
        with col2:
            st.plotly_chart(fig_decision)
        st.divider()
        st.subheader("Emotion to Behavior Mapping")
        st.markdown(f""" **Detected Emotion:** {result["Detected Emotion"]}
                         **Mapped Behavioral State:** {result["Mapped Emotion"]}
                         **Predicted Decision Tendency:** {result["Predicted Decision"]} """)
        st.divider()
        st.subheader("Explanation")
        st.write(result["Explanation"])
        st.divider()
        st.subheader("Behavioral Interpretation")
        st.write(interpretation_text)