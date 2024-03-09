import nltk
from textblob import TextBlob
import spacy
from gensim.models import Word2Vec
import stanza
#v1.0
nltk.download('punkt')
nltk.download('vader_lexicon')
stanza.download('en')  # Download the English language module for CoreNLP

# Initialize objects
nlp_spacy = spacy.load('en_core_web_sm')
nlp_stanza = stanza.Pipeline('en')

def analyze_sentiment_with_nltk(text):
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

def analyze_sentiment_with_textblob(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment

def analyze_sentiment_with_spacy(text):
    doc = nlp_spacy(text)
    sentiment = doc.sentiment
    return sentiment

def analyze_sentiment_with_corenlp(text):
    doc = nlp_stanza(text)
    sentences_sentiment = []
    for sentence in doc.sentences:
        sentiment = sentence.sentiment
        sentences_sentiment.append(sentiment)
    overall_sentiment = sum(sentences_sentiment) / len(sentences_sentiment)
    return overall_sentiment

def analyze_similarity_with_word2vec(text1, text2):
    sentences1 = nltk.sent_tokenize(text1)
    sentences2 = nltk.sent_tokenize(text2)
    model = Word2Vec(sentences1 + sentences2, min_count=1)
    similarity = model.wv.n_similarity(sentences1, sentences2)
    return similarity

def main():
    text = input("Enter the text for sentiment analysis: ")
    print("\nSentiment analysis using NLTK: ")
    print(analyze_sentiment_with_nltk(text))

    print("\nSentiment analysis using TextBlob: ")
    print(analyze_sentiment_with_textblob(text))

    print("\nSentiment analysis using SpaCy: ")
    print(analyze_sentiment_with_spacy(text))

    print("\nSentiment analysis using CoreNLP: ")
    print(analyze_sentiment_with_corenlp(text))

    text2 = input("\nEnter the second text to compare with the first: ")
    print("\nText similarity using Word2Vec: ")
    print(analyze_similarity_with_word2vec(text, text2))

if __name__ == "__main__":
    main()
