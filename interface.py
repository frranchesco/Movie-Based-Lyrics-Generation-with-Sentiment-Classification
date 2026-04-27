import streamlit as st
import pandas as pd
from transformers import pipeline

# Import the functions created
import logic

# 1. PAGE SETUP & CACHING

st.set_page_config(page_title="Movie-to-Lyrics Gen", page_icon="🎬")

st.title("🎬 Movie-Based Lyrics Generator")
st.write("Type a movie name, and we'll analyze its mood to write a custom song!")

@st.cache_data
def load_movie_data():
    # Only loading the necessary columns to save laptop memory
    df = pd.read_csv('top10K-TMDB-movies.csv', usecols=['title', 'overview', 'genre'])
    df = df.dropna(subset=['overview'])
    return df

@st.cache_resource
def load_emotion_model():
    # This prevents the 300MB model from reloading on every keystroke
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )


df_movies = load_movie_data()
emotion_classifier = load_emotion_model()


# 2. USER INTERFACE & LOGIC FLOW

movie_query = st.text_input("Search for a movie:", placeholder="e.g., Inception, The Dark Knight, Toy Story", key="movie_search_box")

if movie_query:
    # 1. Find the movie in the dataset
    match = df_movies[df_movies['title'].str.contains(movie_query, case=False, na=False)]
    
    if not match.empty:
        movie_row = match.iloc[0]
        st.success(f"Found it: **{movie_row['title']}**")
        
        # 2. Run the analysis while showing a loading spinner
        with st.spinner(f"Analyzing the mood and themes of '{movie_row['title']}'..."):
            
            # --- The Logic Pipeline ---
            genre_list = logic.split_genres(movie_row['genre'])
            
            emotions = logic.classify_top_emotions_full_plot(movie_row['overview'], emotion_classifier)
            
            valence = logic.get_weighted_valence(emotions)
            main_mood = logic.assign_main_mood(valence)
            sub_mood = logic.assign_sub_mood(valence)
            
            mood_hints, theme_hints, style_hints = logic.build_genre_profile(genre_list)
            
            prompt = logic.build_lyric_prompt(
                title=movie_row['title'],
                genre_list=genre_list,
                emotions_list=emotions,
                main_mood=main_mood,
                sub_mood=sub_mood,
                theme_hints=theme_hints,
                style_hints=style_hints
            )
            
            # The Output 
            st.divider()
            
            # Display the stats for the user to see
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Main Mood:** {main_mood}")
                st.write(f"**Sub-Mood:** {sub_mood}")
            with col2:
                st.write(f"**Top Emotion:** {emotions[0]['label'].title()} ({emotions[0]['score']:.2f})")
                st.write(f"**Valence Score:** {valence:.2f}")

            
            # Show the prompt
            st.subheader("⚙️ Generated Prompt for the LLM:")
            st.code(prompt, language="text")

            # Final Lyrics
            st.subheader("🎶 Final Lyrics:")
            final_song = logic.generate_final_lyrics(prompt)
            st.write(final_song)
            
            # Analyze generated lyrics sentiment
            lyrics_emotions = logic.classify_top_emotions_full_plot(final_song, emotion_classifier)
            lyrics_valence = logic.get_weighted_valence(lyrics_emotions)
            lyrics_main_mood = logic.assign_main_mood(lyrics_valence)

            st.subheader("🎯 Lyrics Sentiment Analysis")
            st.write(f"Movie Mood: {main_mood}")
            st.write(f"Lyrics Mood: {lyrics_main_mood}")

            if main_mood == lyrics_main_mood:
                st.success("Mood Match")
            else:
                st.warning("Mood differs")
    else:
        st.error("Movie not found in the TMDB dataset. Try another one!")