import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import networkx as nx
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit App
st.title("Text Analysis and Topic Modeling")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # Select the text column for analysis
    text_column = st.selectbox("Select the text column for analysis", df.columns)

    # Preprocess the selected text column
    df['Processed_Text'] = df[text_column].apply(preprocess_text)

    # Step 2: Parameter Selection for LDA
    n_topics = st.slider("Number of Topics for LDA", min_value=2, max_value=10, value=5)
    ngram_range = st.slider("N-gram Range (1-3)", min_value=1, max_value=3, value=(1, 3))

    # Run Analysis
    if st.button("Run Analysis"):
        vectorizer = CountVectorizer(stop_words='english', ngram_range=(ngram_range, ngram_range), max_features=1000)
        term_matrix = vectorizer.fit_transform(df['Processed_Text'])
        feature_names = vectorizer.get_feature_names_out()

        # LDA Topic Modeling
        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_topics = lda_model.fit_transform(term_matrix)

        # Visualization of Connections Between Topics and Key Terms
        def visualize_connections():
            G = nx.Graph()
            for idx, topic in enumerate(lda_model.components_):
                for i in topic.argsort()[:-10 - 1:-1]:
                    G.add_edge(f'Topic {idx+1}', feature_names[i])
            plt.figure(figsize=(12, 8))
            nx.draw(G, with_labels=True, node_color='lightblue', font_size=10, node_size=3000)
            plt.title('Connections Between Topics and Key Terms')
            st.pyplot(plt)

        # Display Results
        st.subheader("Symbol Connections Visualization")
        visualize_connections()
