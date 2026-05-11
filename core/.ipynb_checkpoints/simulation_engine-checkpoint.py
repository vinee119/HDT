import os
import numpy as np
import pandas as pd
import joblib
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
synthetic_model = joblib.load(os.path.join(MODELS_DIR, "synthetic_model.pkl"))
synthetic_encoder = joblib.load(os.path.join(MODELS_DIR, "synthetic_encoder.pkl"))
synthetic_scaler = joblib.load(os.path.join(MODELS_DIR, "synthetic_scaler.pkl"))
scenarios = joblib.load(os.path.join(MODELS_DIR, "scenarios.pkl"))
difficulty_map = joblib.load(os.path.join(MODELS_DIR, "difficulty_map.pkl"))
emotion_names = joblib.load(os.path.join(MODELS_DIR, "goemotions_label_names.pkl"))
goemotion_model = joblib.load(os.path.join(MODELS_DIR, "goemotions_model.pkl"))
vectorizer = joblib.load(os.path.join(MODELS_DIR, "goemotions_vectorizer.pkl"))
emotion_mapping = {"nervousness" : "anxious", "fear" : "anxious", "sadness" : "stressed", "anger" : "stressed", "disappointment" : "stressed", "joy" : "thinking", "optimism" : "thinking", "curiosity" : "thinking", "confusion" : "thinking", "neutral" : "thinking"}
def simulate_behavior(user_text, scenario, difficulty):
    text_vector = vectorizer.transform([user_text])
    emotion_probs = goemotion_model.predict_proba(text_vector)[0]
    dominant_index = np.argmax(emotion_probs)
    dominant_emotion = (emotion_names[dominant_index])
    mapped_emotion = emotion_mapping.get(dominant_emotion, "thinking")
    synthetic_input = pd.DataFrame([{"scenario" : scenario, "difficulty" : difficulty, "emotion" : mapped_emotion, "response_time" : "slow" if difficulty in ["high", "very high"] else "moderate", "personality" : "balanced", "age_group" : "young", "stress_level" : "medium", "context" : "professional"}])
    encoded = synthetic_encoder.transform(synthetic_input)
    scaled = synthetic_scaler.transform(encoded)
    decision = synthetic_model.predict(scaled)[0]
    probabilities = synthetic_model.predict_proba(scaled)[0]
    confidence = np.max(probabilities)
    explanation = (f"The dominant detected emotion is '{dominant_emotion}', " f"which maps to the behavioral state '{mapped_emotion}'.\n" f"For the selected scenario '{scenario}' with difficulty level '{difficulty}', " f"the predicted decision tendency is '{decision}'.")
    return {"Predicted Decision" : decision, "Confidence" : round(float(confidence), 3), "Detected Emotion" : dominant_emotion, "Mapped Emotion" : mapped_emotion, "Explanation" : explanation}
def behavioral_interpretation(mapped_emotion, difficulty, decision):
    interpretation = []
    if mapped_emotion in ["anxious", "stressed", "panic", "overwhelmed"]:
        interpretation.append("Feeling anxious in challenging situations can increase the tendency to avoid decisions.")
    elif mapped_emotion in ["calm", "focused"]:
        interpretation.append("A calm emotional state usually supports clear and balanced decision-making.")
    elif mapped_emotion in ["concerned", "thinking"]:
        interpretation.append("A reflective emotional state may lead to careful and cautious decisions.")
    else:
        interpretation.append(f"The emotional state '{mapped_emotion}' influences how decisions are made under situational demands.")
    if difficulty in ["high", "very high"]:
        interpretation.append("Higher difficulty levels can increase emotional pressure and effect decision patterns.")
    elif difficulty == "medium":
        interpretation.append("Moderate difficulty allows better control over emotional reactions.")
    else:
        interpretation.append("Lower difficulty situations reduce emotional pressure on decisions.")
    if decision == "avoidant":
        interpretation.append("Using structured planning and breaking the problem into smaller steps may help reduce avoidance.")
    elif decision == "impulsive":
        interpretation.append("Taking a short pause before deciding may improve decision quality.")
    elif decision == "analytical":
        interpretation.append("Continuing structured analysis can support consistent decision outcomes.")
    elif decision == "cautious":
        interpretation.append("Balancing caution with timely action may improve results.")
    else:
        interpretation.append("Improving emotional awareness may enhance decision effectiveness.")
    return " ".join(interpretation)