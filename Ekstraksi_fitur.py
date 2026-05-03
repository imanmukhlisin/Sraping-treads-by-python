from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, df, text_col='commentText'):
        X = self.vectorizer.fit_transform(df[text_col])
        return X

    def transform(self, df, text_col='commentText'):
        return self.vectorizer.transform(df[text_col])