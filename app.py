import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import streamlit as st
import emoji
import pickle
from scipy.sparse import hstack
from PIL import Image
import pandas as pd


import matplotlib.pyplot as plt
from textblob import TextBlob

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Import the library

# Download NLTK resources (run this once)
nltk.download('stopwords')
nltk.download('punkt')

# Load the TF-IDF vectorizer from the pickle file
tfidf_vectorizer_filename = 'tfidf_vectorizer1.pkl'  # Update the path accordingly
with open(tfidf_vectorizer_filename, 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Load the pre-trained machine learning model using pickle
model_filename = 'SVMEXMODEL2.sav'  # Adjust the filename if needed
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Flag to check if TF-IDF vectorizer is fitted
is_tfidf_fitted = False

def has_emoji(text):
    # Check if the text contains emojis using the emoji library
    return bool(emoji.get_emoji_regexp().search(text))

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters, URLs, and user mentions
    text = re.sub(r'http\S+|www\S+|https\S+|@[^\s]+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)

    # Remove emojis using a regular expression
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    # Remove emojis using the regex pattern
    text = emoji_pattern.sub(r'', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])

    # Remove remaining non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    return text

# Define your pages as functions with additional styling
def home_page():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
        .custom-title {
            font-family: 'Helvetica', sans-serif;
            font-size: 36px;
            color: #000000;  /* Change to your preferred color */
            font-weight: bold;  /* Add this line for bold text */
        }
    </style>
""", unsafe_allow_html=True)

    st.markdown('<p class="custom-title">Depression Detector</p>', unsafe_allow_html=True)

    st.info("Welcome to the Depression Detector App! Here, you'll be able to learn more on this social issue as well as identify if what you say may contain possible depression. Explore the app to understand more on this project! ")
    st.markdown("### Did you know?")
    st.markdown("Close to 280 million people worldwide suffer from depression. That's an approximate 3.9% of the world's population. This causes a negative impact to lives of individuals in their work environment, relationships and also daily productivity.")

    st.markdown("### How about Malaysia?")
    st.markdown("Studies have shown 2.3% of Malaysians adults are facing the highest rates of depression with young adults (ages 18-29) spearheading the leaderboard. ")
    
    
    st.markdown("### Current Trend")
    st.markdown("Individuals now prefer expressing themselves on social media by sharing their opinions and feelings on various topics. This acts as a platform for data to be collected and analyzed which is later used to build a system that can determine the presence of depression within a text.") 
    #image_path1 = 'slide chart.jpg'
    #st.image(image_path1, width=650)

    #st.info("We can observe that a higher usage of social media correlates with individuals containing higher depressive symptoms. With that, this project aims to utilise machine learning to identify signs of depression in social media by analyzing inputed text from user")

    st.success("""
       Go to the next page to test out the Depression Detector!
    """)



def depDetect_page():
    st.title("📊 Depression Detector")
    st.success("1. Firstly, insert a text you wish to analyze\n2. Then, click the 'Try it out' button\n3. Your result will be displayed below")

    user_text = st.text_area("What do you wish to say? ", max_chars=250, height=150)

    if st.button("Try it out"):
        # Preprocess user text
        cleaned_text = clean_text(user_text)

        # Perform TF-IDF vectorization
        test_text = tfidf_vectorizer.transform([cleaned_text])

        analyzer = SentimentIntensityAnalyzer()

        # Analyze sentiment using VADER
        sentiment_scores = analyzer.polarity_scores(cleaned_text)

        # Display sentiment scores
        st.write("Sentiment Scores:", sentiment_scores)

        # Determine sentiment category based on compound score
        sentiment_polarity = sentiment_scores['compound']

        if sentiment_polarity > 0:
            sentiment_category = "Positive"
        elif sentiment_polarity < 0:
            sentiment_category = "Negative"
        else:
            sentiment_category = "Neutral"

        # Display sentiment category
        st.write(f"Sentiment Category: {sentiment_category}")

        # Create a bar chart for visualization
        fig, ax = plt.subplots()
        ax.bar(["Positive", "Neutral", "Negative"],
               [max(0, sentiment_polarity), 0, abs(min(0, sentiment_polarity))],
               color=['green', 'gray', 'red'])
        ax.set_ylim(0, 1)  # Set y-axis limit to 1 for better visualization
        ax.set_title("Sentiment Analysis Chart")

        # Display the chart in Streamlit
        st.pyplot(fig)

        # Perform prediction with the loaded model
        result = loaded_model.predict(test_text)
        if result == 0:
            st.warning("Assesment Result : Depression detected.")
            
        else:
            st.success("Assesment Result : No depression detected.")
           
def eda_page():
    st.title("Exploratory Data Analysis")
    st.info("The charts below depict the exploratory data analysis observed in this project")

    tabs = ["Depression Analysis", "Non-Depression Analysis", "Positive Sentiment", "Negative Sentiment", "Neutral Sentiment", "Sentiment Distribution"]
    selected_tab = st.selectbox("Explore the analysis!", tabs)



    if selected_tab == "Depression Analysis":
        # WordCloud for Depression Analysis
        st.markdown("### WordCloud for Depression tagged words")
        wordcloud_image_path = 'newDepClou.png'  # Update with the correct file path
        wordcloud_image = Image.open(wordcloud_image_path)
        max_size = (400, 200)  # Adjust the size as needed
        wordcloud_image.thumbnail(max_size)
        st.image(wordcloud_image, use_column_width=True)
        st.markdown("""
        This WordCloud visualizes the most frequent words in tweets labeled for depression. 
        The size of each word represents its frequency in the dataset. It can be observed that words such as 'got', 'today' and 'work' are having one of the highest frequencies.
        """)

        st.markdown("### Random Words Associated With Depression ")
        another_image_path = 'Deprandom_keywords_chart.png'  # Update with the correct file path
        another_image = Image.open(another_image_path)
        max_size_another = (600, 400)  # Adjust the size as needed
        another_image.thumbnail(max_size_another)
        st.image(another_image, use_column_width=True)
        st.markdown("""
        The chart above displays random words from the dataset that are Depression tagged. Words such as 'know', 'thing' and 'feel' seem to be standing out which indicates a focus on cognitive and emotional elements such as reference to thought, mental processes and also emotions.
        """)

        st.markdown("### Sample Text ")
        st.markdown(" Sample : 'is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!' ")

    if selected_tab == "Non-Depression Analysis":
        # WordCloud for Non-Depression Analysis
        st.markdown("### WordCloud for Non-Depression tagged words")
        wordcloud_image_path = 'newNoDepclou.png'  # Update with the correct file path
        wordcloud_image = Image.open(wordcloud_image_path)
        max_size = (400, 200)  # Adjust the size as needed
        wordcloud_image.thumbnail(max_size)
        st.image(wordcloud_image, use_column_width=True)
        st.markdown("""
        This WordCloud visualizes the most frequent words in tweets labelled for Non-Depression.
        The size of each word represents its frequency in the dataset. It can be observed that words such as 'thank', 'love' and 'time' are having one of the highest frequencies.
        """)

        st.markdown("### Random Words Associated With Non-Depression ")
        another_image_path2 = 'random_keywords_chart (1).png'  # Update with the correct file path
        another_image2 = Image.open(another_image_path2)
        max_size_another2 = (600, 400)  # Adjust the size as needed
        another_image2.thumbnail(max_size_another2)
        st.image(another_image2, use_column_width=True)
        st.markdown("""
        The chart above displays random words from the dataset that are Non-Depression tagged. Words such as 'going', 'time' and 'lol' seem to have relatively higher frequencies which indicates a positive and light atmosphere. For instance, 'Going' suggest's optimism as it is linked with carrying out an activity or plan.
        """)

        st.markdown("### Sample Text ")
        st.markdown(" Sample : 'Good morning people of twitter. TGIFriday! Thats wassssup! Today is going to be a hellofagood day!' ")
                    
    if selected_tab == "Positive Sentiment":
        # WordCloud for Positive Sentiment
        st.markdown("### WordCloud for Positive Sentiment")
        wordcloud_image_path = 'wcloudNewPos.png'  # Update with the correct file path
        wordcloud_image = Image.open(wordcloud_image_path)
        max_size = (400, 200)  # Adjust the size as needed
        wordcloud_image.thumbnail(max_size)
        st.image(wordcloud_image, use_column_width=True)
        st.markdown("""
        This WordCloud visualizes the most frequent words in the dataset annoted as a positive sentiment. 
        The size of each word represents its frequency in the positive labelled tweets.
        """)

    elif selected_tab == "Negative Sentiment":
        # WordCloud for Negative Sentiment
        st.markdown("### WordCloud for Negative Sentiment")
        negative_wordcloud_image_path = 'wcloudNewNega.png'  # Update with the correct file path
        negative_wordcloud_image = Image.open(negative_wordcloud_image_path)
        max_size = (400, 200)  # Adjust the size as needed
        negative_wordcloud_image.thumbnail(max_size)
        st.image(negative_wordcloud_image, use_column_width=True)
        st.markdown("""
        This WordCloud visualizes the most frequent words in tweets annoted as a negative sentiment. 
        The size of each word represents its frequency in the negative labelled tweets.
        """)

    elif selected_tab == "Neutral Sentiment":
        # WordCloud for Neutral Sentiment
        st.markdown("### WordCloud for Neutral Sentiment")
        neutral_wordcloud_image_path = 'wcloudNewNeutr.png'  # Update with the correct file path
        neutral_wordcloud_image = Image.open(neutral_wordcloud_image_path)
        max_size = (450, 200)  # Adjust the size as needed
        neutral_wordcloud_image.thumbnail(max_size)
        st.image(neutral_wordcloud_image, use_column_width=True)
        st.markdown("""
        This WordCloud visualizes the most frequent words in tweets annoted as a neutral sentiment. 
        The size of each word represents its frequency in the neutral labelled tweets.
        """)

    elif selected_tab == "Sentiment Distribution":
        # Sentiment Distribution
        st.markdown("### Distribution of Sentiments")
        image_pathSent = 'sentiment_category_distribution.png'
        imageSent = Image.open(image_pathSent)
        st.image(imageSent, use_column_width=True)
        st.markdown("""
        This chart represents the distribution of sentiment categories in the dataset. 
        The breakdown of each category is displayed above each bar plot. 
        The addition of the 3 categories sum up to 120,000.
        """)


def docu_page():
    st.title("Documentation")

    st.markdown("<h5>Home</h5>", unsafe_allow_html=True)
    st.info("The home page displays the introduction and some facts related to depression and the project. ")
   
    st.markdown("<h5>Depression Detector</h5>", unsafe_allow_html=True)
    st.info("This page contains a trained machine learning model which is used to detect the presence of depression based on user input text. User can enter text and identify the result based on the steps shown in the page.")
    
    st.markdown("<h5>Exploratory Data Analysis</h5>", unsafe_allow_html=True)
    st.info("The EDA page will display all the relevant analysis in accordance to this project such as understanding the words associated with depression and non-depression as well as the overview of the sentiments from the text.")
   
    st.markdown("<h5>About</h5>", unsafe_allow_html=True)
    st.info("This page contains information about the creator of this web app as well as more information about the project.")

    # Provide links and information for support

def about_page():
    st.title("About")

    tabs2 = ["The Creator", "The Project"]
    selected_tab2 = st.selectbox("Learn more about : ", tabs2)


    if selected_tab2 == "The Creator":
        st.markdown("<h5>Jared Thomas</h5>", unsafe_allow_html=True)
        image_path66 = 'jtk pic crop.jpg'
        st.image(image_path66, width=250)
        info_text = "Greetings! I'm Jared, a 3rd Year Data Science student from the University Of Malaya. Fueled by curiosity, I embark on a continuous journey of exploration, delving into new findings day in, day out. I believe that knowledge is power and this drives me to continuously learn and push myself beyond boundaries. Check out my [LinkedIn](https://www.linkedin.com/in/jared-thomas-615abb234/) & [Github](https://github.com/dashboard)"

        # Display the modified text with hyperlinks
        st.markdown(info_text, unsafe_allow_html=True)

    elif selected_tab2 == "The Project":
        # Sentiment Distribution
        st.markdown("### Depression Detector")
        st.info("This project utilises sentiment analysis as an additional feature for model training which is later used as the backend engine to identify if a user's inputed text is positive for depression. User may input a text of their choice and observe the model's analysis on it. ")

        st.markdown("### Dataset used")
        info_text4 = "The dataset used is sourced from Kaggle. It contains 1,048,576 rows, but only 60,000 rows were randomly sampled for each category being depression and non-depression."

    # Display the modified text with a hyperlink
        st.markdown(info_text4, unsafe_allow_html=True)

        image_path67 = 'LabelNew.png'
        st.image(image_path67, width=600)

        st.markdown("### Feature Extraction & Engineering")
        st.markdown("Term Frequency-Inverse Document Frequency (TFIDF) - To identify the importance of a word to it's document. N-grams (Unigrams) were also used. Sentiment analysis was also carried out where sentiments of tweets were determined. Then, TFIDF with Unigram is combined with features generated from sentiment analysis and fed to the model for training and testing.")

        st.markdown("### Machine Learning Model")
        st.markdown("As this is a classification problem, the model used is Support Vector Machine. After carrying out modelling with numerous ML algorithms, it was reported with Support Vector Machine having the best outcome. ")

        st.markdown("##### Model Performance")
        image_path80 = 'results.png'
        st.image(image_path80, width=450)


        
st.set_page_config(layout="wide")

# Customize tab font size and style
st.markdown("""
<style>
div[role="tablist"] > div {
    font-size: 18px !important;  /* Adjust the font size as needed */
    font-weight: bold !important; /* Add bold style if desired */
}
</style>
""", unsafe_allow_html=True)

# Create a wider container for tabs with a light blue background
st.markdown("""
<style>
.stTabs {
  background-color: #F0F8FF;  /* Light blue background color for wider container */
  padding: 20px;
  border-radius: 10px;  /* Adjust border radius as needed */
}
</style>
""", unsafe_allow_html=True)

# Create a wider container for tabs
tabs = st.tabs(["Home 🏚️", "Depression Detector 🫥", "Exploratory Data Analysis 📈", "Documentation 🗺️", "About ©️"])

with tabs[0]:
    home_page()
with tabs[1]:
    depDetect_page()
with tabs[2]:
    eda_page()
with tabs[3]:
    docu_page()
with tabs[4]:
    about_page()


