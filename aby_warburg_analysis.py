import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import nltk
from nltk.corpus import stopwords
from collections import Counter

# Download NLTK stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Streamlit App
st.title("Aby Warburg-inspired Word Analysis Tool")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # Select the text column for analysis
    text_column = st.selectbox("Select the text column for analysis", df.columns)

    # Step 2: Preprocess and Analyze Text
    if st.button("Analyze Words"):
        # Combine all text into a single string
        all_text = ' '.join(df[text_column].dropna().astype(str))

        # Tokenize the text using split (basic whitespace tokenization)
        words = all_text.lower().split()

        # Remove stopwords and non-alphabetic tokens
        words = [word for word in words if word.isalpha() and word not in stop_words]

        # Count word frequencies
        word_counts = Counter(words)

        # Display the most common words
        common_words = word_counts.most_common(20)
        st.write("Most Common Words:", common_words)

        # Step 3: Visualize Connections Between Words
        st.subheader("Word Connection Visualization")
        
        # Create a graph from word pairs
        G = nx.Graph()
        for i in range(len(words) - 1):
            G.add_edge(words[i], words[i + 1])
        
        plt.figure(figsize=(10, 8))
        nx.draw_networkx(G, with_labels=True, node_size=20, font_size=10, node_color='skyblue', edge_color='gray')
        plt.title('Connections Between Words')
        st.pyplot(plt)
