import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class TextPreprocessor:
    def __init__(self):
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopwords = set(StopWordRemoverFactory().get_stop_words())

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in self.stopwords])
        text = self.stemmer.stem(text)
        return text

    def transform(self, df, text_col='commentText'):
        df[text_col] = df[text_col].astype(str).apply(self.preprocess)
        return df