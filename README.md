Based on the final state of your code and the integration of the OpenAI lyric generator, here is a professional and updated README for your GitHub repository.

Movie-Based Lyrics Generation with Sentiment Classification
Project Description
We built an AI-powered application that creates original song lyrics by analyzing the emotional and thematic depth of movies. The system uses a multi-stage machine learning pipeline to transform a movie's plot summary into a detailed creative prompt for lyric generation.

How it Works

Movie Search: The user enters a movie title.

Plot Retrieval: The application retrieves the movie’s plot overview and genre information from a curated TMDB dataset.

Automatic Sentiment Analysis: Using a distilroberta-base emotion classification model, the app automatically analyzes the plot to extract primary emotions (e.g., joy, anger, sadness) and calculates a weighted valence score to determine the overall mood.

Thematic Profiling: The system maps movie genres to specific musical "hints" (moods, themes, and styles) to ensure the generated lyrics match the cinematic vibe.

Lyric Generation: A detailed profile containing the movie title, emotional blend, and style hints is sent to the OpenAI API (GPT-4o-mini) to generate cinematic, emotionally expressive lyrics.

Project Structure

interface.py: The Streamlit-based user interface that handles user input and displays the emotional analysis and final lyrics.

logic.py: The application’s "brain" containing the sentiment classification models, valence math, and API integration logic.

top10K-TMDB-movies.csv: A dataset of 10,000 top-rated movies used for plot and genre retrieval. https://www.kaggle.com/datasets/ahsanaseer/top-rated-tmdb-movies-10k 

Setup and Demo Instructions

Install required libraries: pip install streamlit transformers torch pandas openai.

Ensure you have a valid OpenAI API Key.

Add your API key to the OPENAI_API_KEY variable in logic.py.

Run the application: streamlit run interface.py.
