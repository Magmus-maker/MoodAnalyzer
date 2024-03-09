# Text Mood Analyzer

Text Mood Analyzer is a Python application that analyzes the sentiment and similarity of textual data using various natural language processing libraries such as NLTK, TextBlob, SpaCy, Gensim, and CoreNLP via the stanza library.

## Features

- Sentiment analysis using NLTK, TextBlob, SpaCy, and CoreNLP.
- Text similarity comparison using Word2Vec from Gensim.
- Easy-to-use command-line interface.
- Supports multiple natural language processing techniques for accurate sentiment analysis and text similarity assessment.

## Requirements

- Python 3.x
- NLTK
- TextBlob
- SpaCy
- Gensim
- stanza

## Installation
Clone this repository:
```
git clone https://github.com/your-username/text-mood-analyzer.git
```

Install the required dependencies:
```pip install -r requirements.txt```

Download the necessary NLTK corpora:
```python -m nltk.downloader punkt```
```python -m nltk.downloader vader_lexicon```

Download the English language model for CoreNLP using stanza:
```python -m stanza.download en```

Usage
Run the application:
python main.py
Follow the prompts to enter the text for sentiment analysis and text similarity comparison.


License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
This project was inspired by the need for a simple yet effective sentiment analysis tool.
Special thanks to the developers of NLTK, TextBlob, SpaCy, Gensim, and CoreNLP for their invaluable contributions to the field of natural language processing.
