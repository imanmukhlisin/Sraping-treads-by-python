from Scraping import TikTokScraper
from Processing import TextPreprocessor
from Labeling import SentimentLabeler
from Ekstraksi_fitur import FeatureExtractor
from Pelatihan import SentimentModel
from Evaluasi import Evaluator
from Visualisasi import Visualizer
from n_gram import NgramAnalyzer

# 1. Scraping data TikTok via Apify
api_token = os.getenv("APIFY_TOKEN")  # Ganti dengan token kamu
video_urls = [
    "https://www.tiktok.com/@gendisptrii/video/7081268229422796058",
]
scraper = TikTokScraper(apify_token)
df = scraper.scrape(video_urls, output_csv="data_tiktok.csv")

# 2. Pra-pemrosesan
preprocessor = TextPreprocessor()
df = preprocessor.transform(df)

# 3. Labeling sentimen
labeler = SentimentLabeler()
df = labeler.transform(df)
df.to_csv("data_tiktok_labeled.csv", index=False, encoding="utf-8")  # Simpan hasil labeling

# 4. Ekstraksi fitur
extractor = FeatureExtractor()
X = extractor.fit_transform(df)
y = df['sentiment']

# 5. Pelatihan model
model = SentimentModel()
model.train(X, y)

# 6. Evaluasi model
y_pred = model.predict(X)
evaluator = Evaluator()
evaluator.evaluate(y, y_pred)

# Simpan hasil prediksi
df['predicted_sentiment'] = y_pred
df.to_csv("data_tiktok_predicted.csv", index=False, encoding="utf-8")

# 7. Visualisasi
visualizer = Visualizer()
visualizer.plot_sentiment_distribution(df)
visualizer.plot_wordcloud(df)
for sent in ['positif', 'negatif', 'netral']:
    visualizer.plot_wordcloud(df, sentiment=sent)

# 8. Analisis n-gram trigram per sentimen
ngram_analyzer = NgramAnalyzer(folder="hasil")
for sent in ['positif', 'netral', 'negatif']:
    ngram_analyzer.plot_top_ngrams(df, sentiment=sent, ngram_range=(3,3), top_n=10)

print("Pipeline selesai. Cek hasil di folder ini.")