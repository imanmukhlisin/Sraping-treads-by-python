import pandas as pd

class SentimentLabeler:
    def __init__(self):
        # Dummy: Ganti dengan model/aturan labeling otomatis/manual
        self.positive_words = ['suka', 'bagus', 'keren', 'hebat']
        self.negative_words = ['buruk', 'jelek', 'tidak', 'benci']

    def label(self, text):
        if any(word in text for word in self.positive_words):
            return 'positif'
        elif any(word in text for word in self.negative_words):
            return 'negatif'
        else:
            return 'netral'

    def transform(self, df, text_col='commentText'):
        df['sentiment'] = df[text_col].apply(self.label)
        return df