import pandas as pd
from openai import OpenAI

# 1. DICTIONARIES & MAPPINGS

emotion_to_valence = {
    "anger": 1.3,
    "fear": 1.6,
    "sadness": 1.8,
    "neutral": 3.0,
    "surprise": 3.6,
    "joy": 4.4
}

genre_profile_map = {
    "Action": {
        "mood_hints": ["intense", "urgent", "energetic"],
        "theme_hints": ["conflict", "survival", "revenge"],
        "style_hints": ["dramatic", "driving", "bold"]
    },
    "Adventure": {
        "mood_hints": ["uplifting", "restless", "hopeful"],
        "theme_hints": ["journey", "discovery", "courage"],
        "style_hints": ["expansive", "cinematic", "vivid"]
    },
    "Animation": {
        "mood_hints": ["playful", "bright", "expressive"],
        "theme_hints": ["growth", "friendship", "imagination"],
        "style_hints": ["colorful", "light", "whimsical"]
    },
    "Comedy": {
        "mood_hints": ["light", "playful", "cheerful"],
        "theme_hints": ["chaos", "relationships", "fun"],
        "style_hints": ["witty", "bouncy", "clever"]
    },
    "Crime": {
        "mood_hints": ["dark", "tense", "gritty"],
        "theme_hints": ["betrayal", "power", "guilt"],
        "style_hints": ["moody", "sharp", "gritty"]
    },
    "Drama": {
        "mood_hints": ["reflective", "serious", "emotional"],
        "theme_hints": ["loss", "relationships", "growth"],
        "style_hints": ["poetic", "grounded", "emotional"]
    },
    "Family": {
        "mood_hints": ["warm", "safe", "hopeful"],
        "theme_hints": ["love", "belonging", "togetherness"],
        "style_hints": ["gentle", "heartfelt", "accessible"]
    },
    "Fantasy": {
        "mood_hints": ["wonder", "epic", "dreamlike"],
        "theme_hints": ["destiny", "magic", "sacrifice"],
        "style_hints": ["mythic", "lush", "imaginative"]
    },
    "History": {
        "mood_hints": ["solemn", "reflective", "grave"],
        "theme_hints": ["legacy", "struggle", "change"],
        "style_hints": ["formal", "evocative", "weighty"]
    },
    "Horror": {
        "mood_hints": ["ominous", "uneasy", "dark"],
        "theme_hints": ["fear", "death", "survival"],
        "style_hints": ["haunting", "shadowy", "tense"]
    },
    "Music": {
        "mood_hints": ["expressive", "passionate", "uplifting"],
        "theme_hints": ["performance", "dreams", "identity"],
        "style_hints": ["lyrical", "rhythmic", "emotive"]
    },
    "Mystery": {
        "mood_hints": ["uncertain", "tense", "curious"],
        "theme_hints": ["secrets", "truth", "deception"],
        "style_hints": ["layered", "cryptic", "atmospheric"]
    },
    "Romance": {
        "mood_hints": ["tender", "heartfelt", "intimate"],
        "theme_hints": ["love", "longing", "heartbreak"],
        "style_hints": ["personal", "soft", "emotional"]
    },
    "Science Fiction": {
        "mood_hints": ["atmospheric", "uncertain", "expansive"],
        "theme_hints": ["identity", "technology", "isolation"],
        "style_hints": ["cinematic", "futuristic", "reflective"]
    },
    "TV Movie": {
        "mood_hints": ["direct", "accessible", "contained"],
        "theme_hints": ["relationships", "conflict", "resolution"],
        "style_hints": ["clear", "simple", "focused"]
    },
    "Thriller": {
        "mood_hints": ["tense", "urgent", "suspenseful"],
        "theme_hints": ["danger", "betrayal", "pressure"],
        "style_hints": ["tight", "edgy", "dramatic"]
    },
    "War": {
        "mood_hints": ["grave", "intense", "somber"],
        "theme_hints": ["loss", "courage", "trauma"],
        "style_hints": ["solemn", "powerful", "raw"]
    },
    "Western": {
        "mood_hints": ["rugged", "lonely", "stoic"],
        "theme_hints": ["justice", "freedom", "survival"],
        "style_hints": ["dusty", "sparse", "mythic"]
    }
}

# 2. DATA CLEANING & FORMATTING

def split_genres(text):
    if pd.isna(text):
        return []
    parts = text.split(',')
    cleaned = []
    for g in parts:
        g = g.strip()
        if g != "":
            cleaned.append(g)
    return cleaned

def split_text_into_chunks(text, chunk_size=400, overlap=50):
    if pd.isna(text) or str(text).strip() == "":
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        start = start + (chunk_size - overlap)
    return chunks


# 3. AI EMOTION ANALYSIS

def classify_top_emotions_full_plot(text, emotion_classifier, top_n=3):
    """
    Analyzes the movie overview. 
    Notice we pass 'emotion_classifier' as an argument so the UI can cache it!
    """
    if pd.isna(text) or str(text).strip() == "":
        return []

    chunks = split_text_into_chunks(text, chunk_size=400, overlap=50)
    emotion_totals = {}

    for chunk in chunks:
        results = emotion_classifier(chunk, truncation=True)
        if len(results) > 0 and isinstance(results[0], list):
            results = results[0]

        for item in results:
            label = item["label"]
            score = item["score"]
            if label == "disgust":
                continue
            if label not in emotion_totals:
                emotion_totals[label] = 0
            emotion_totals[label] += score

    averaged_results = []
    num_chunks = len(chunks) if len(chunks) > 0 else 1

    for label in emotion_totals:
        avg_score = emotion_totals[label] / num_chunks
        averaged_results.append({
            "label": label,
            "score": avg_score
        })

    def sort_by_score(item):
        return item["score"]

    averaged_results = sorted(averaged_results, key=sort_by_score, reverse=True)
    return averaged_results[:top_n]

# 4. VALENCE & MOOD SCORING

def get_weighted_valence(emotions_list):
    weighted_sum = 0
    total_score = 0

    for item in emotions_list:
        emotion = item["label"]
        score = item["score"]
        
        if emotion in emotion_to_valence and pd.notna(score):
            weighted_sum += emotion_to_valence[emotion] * score
            total_score += score

    if total_score == 0:
        return None
    return weighted_sum / total_score

def assign_main_mood(valence):
    if valence is None or pd.isna(valence):
        return None
    if 1.0 <= valence <= 2.4:
        return "Negative"
    elif 2.5 <= valence <= 3.4:
        return "Neutral/Mixed"
    elif 3.5 <= valence <= 5.0:
        return "Positive"
    else:
        return None

def assign_sub_mood(valence):
    if valence is None or pd.isna(valence):
        return None
    if 1.0 <= valence <= 1.6:
        return "Despair"
    elif 1.7 <= valence <= 2.0:
        return "Sad"
    elif 2.1 <= valence <= 2.4:
        return "Melancholic"
    elif 2.5 <= valence <= 2.8:
        return "Uncertain"
    elif 2.9 <= valence <= 3.1:
        return "Reflective"
    elif 3.2 <= valence <= 3.4:
        return "Hopeful-Neutral"
    elif 3.5 <= valence <= 3.9:
        return "Optimistic"
    elif 4.0 <= valence <= 4.4:
        return "Happy"
    elif 4.5 <= valence <= 5.0:
        return "Excited"
    else:
        return None

def build_genre_profile(genre_list):
    mood_hints = []
    theme_hints = []
    style_hints = []

    for genre in genre_list:
        if genre in genre_profile_map:
            profile = genre_profile_map[genre]
            mood_hints.extend(profile["mood_hints"])
            theme_hints.extend(profile["theme_hints"])
            style_hints.extend(profile["style_hints"])

    # remove duplicates while keeping order
    unique_mood_hints = list(dict.fromkeys(mood_hints))
    unique_theme_hints = list(dict.fromkeys(theme_hints))
    unique_style_hints = list(dict.fromkeys(style_hints))

    return unique_mood_hints, unique_theme_hints, unique_style_hints

# 5. PROMPT GENERATOR

def build_lyric_prompt(title, genre_list, emotions_list, main_mood, sub_mood, theme_hints, style_hints):
    genres = ", ".join(genre_list)
    themes = ", ".join(theme_hints[:4]) if theme_hints else ""
    styles = ", ".join(style_hints[:3]) if style_hints else ""
    
    # Format the emotions list into a clean string
    emotions_str = ", ".join([item['label'] for item in emotions_list])

    prompt = f"""
Write original song lyrics inspired by the following movie profile.

Movie title: {title}
Genres: {genres}
Primary emotional blend: {emotions_str}
Main mood: {main_mood}
Sub-mood: {sub_mood}
Theme hints: {themes}
Style hints: {styles}

Instructions:
- Write emotionally expressive and cinematic lyrics
- Reflect the main mood, sub-mood, and themes
- Use vivid imagery
- Do not mention the movie title directly
- Structure the output as Verse 1, Chorus, Verse 2, Chorus
"""
    return prompt.strip()

# 6. FINAL LYRIC GENERATOR

OPENAI_API_KEY = ""  # Tester should paste their own OpenAI API key here

def generate_final_lyrics(prompt):
    """
    REAL FUNCTION: Calls OpenAI to generate lyrics based on the movie profile.
    """
    client = OpenAI(api_key=OPENAI_API_KEY) #

    response = client.chat.completions.create(
        model="gpt-4o-mini", # Standard, cost-effective model
        messages=[
            {"role": "system", "content": "You are a creative songwriter."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content.strip() #