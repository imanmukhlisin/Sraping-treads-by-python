import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class NgramAnalyzer:
    def __init__(self, folder="hasil"):
        self.folder = folder
        stop_factory = StopWordRemoverFactory()
        self.stopwords = stop_factory.get_stop_words()

    def plot_top_ngrams(self, df, sentiment, ngram_range=(3,3), top_n=10):
        text = df[df['sentiment'] == sentiment]['commentText'].astype(str)
        vec = CountVectorizer(ngram_range=ngram_range, stop_words=self.stopwords)
        X = vec.fit_transform(text)
        sum_words = X.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]
        if not words_freq:
            print(f"Tidak ada n-gram untuk sentimen {sentiment}")
            return
        words, counts = zip(*words_freq)
        plt.figure(figsize=(10,6))
        plt.barh(words, counts, color=plt.cm.Set3.colors)
        plt.xlabel("Jumlah Kemunculan")
        plt.ylabel("Trigram")
        plt.title(f"Top {top_n} Trigram - Sentimen {sentiment.capitalize()}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{self.folder}/top{top_n}_trigram_{sentiment}.png")
        plt.close()