import string
from collections import Counter

import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

text = open('read.txt', encoding='utf-8').read()

lower_case = text.lower()
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

#tokenized_words = cleaned_text.split()
tokenized_words = word_tokenize(cleaned_text,"english")

"""
stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
"""

final_words = []
for word in tokenized_words:
    if word not in stopwords.words("english"):
        final_words.append(word)


emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
        word, emotion = clear_line.split(':')

        if word in final_words:
            emotion_list.append(emotion)


w = Counter(emotion_list)

def sentiment_analyse(sentiment_text):
    sentiment_score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = sentiment_score['neg']
    pos = sentiment_score['pos']
    if neg > pos:
        print("Negative Sentiment")
    elif pos > neg:
        print("Positive Sentiment")
    else:
        print("Neutral Text")

sentiment_analyse(cleaned_text)

plt.pie(w.values(),labels = w.keys())
plt.savefig('graph.png')
plt.show()