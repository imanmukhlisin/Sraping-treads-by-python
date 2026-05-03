import matplotlib.pyplot as plt
from wordcloud import WordCloud

class Visualizer:
    def plot_sentiment_distribution(self, df):
        sent_count = df['sentiment'].value_counts()
        sent_count.plot(kind='bar', color=['green', 'grey', 'red'])
        plt.title("Distribusi Sentimen Komentar")
        plt.xlabel("Sentimen")
        plt.ylabel("Jumlah")
        plt.tight_layout()
        plt.savefig("sentimen_dist.png")
        plt.close()

    def plot_wordcloud(self, df, sentiment=None):
        if sentiment:
            text = ' '.join(df[df['sentiment'] == sentiment]['commentText'].astype(str))
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            wc.to_file(f"wordcloud_{sentiment}.png")
        else:
            text = ' '.join(df['commentText'].astype(str))
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            wc.to_file("wordcloud_all.png")