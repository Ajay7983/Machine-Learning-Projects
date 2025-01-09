import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
nltk.download('punkt')

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load and Clean Data:

@st.cache_data
def load_data():
    return pd.read_excel('C:/Users/Ajay/Downloads/Machine learning capstone project/Capstone project 4 Ajay/Womens Clothing Reviews Data.xlsx')

df = load_data()

# Create Visualizations and Widgets:

st.title('Customer Review Analysis Dashboard')

# Show basic statistics
st.header('Basic Statistics')
st.write(df.describe())

# Age Distribution
st.header('Age Distribution')
plt.figure(figsize=(10, 6))
sns.histplot(df['Customer Age'], bins=20, kde=True)
st.pyplot()

# Location Distribution
st.header('Location Distribution')
location_counts = df['Location'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=location_counts.index, y=location_counts.values)
plt.xticks(rotation=90)
st.pyplot()

# Rating Distribution
st.header('Rating Distribution')
plt.figure(figsize=(10, 6))
sns.histplot(df['Rating'], bins=10, kde=True)
st.pyplot()

# Word Cloud
st.header('Word Cloud of Review Texts')
text = ' '.join(review for review in df['Review Text'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot()

# Sentiment Analysis
st.header('Sentiment Analysis')
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['Sentiment'] = df['Review Text'].dropna().apply(get_sentiment)
sentiment_mean = df.groupby(['Channel'])['Sentiment'].mean()
st.bar_chart(sentiment_mean)
