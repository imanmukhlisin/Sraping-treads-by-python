from sklearn.naive_bayes import MultinomialNB

class SentimentModel:
    def __init__(self):
        self.model = MultinomialNB()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)