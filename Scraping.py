import pandas as pd
from apify_client import ApifyClient

class TikTokScraper:
    def __init__(self, apify_token):
        self.client = ApifyClient(apify_token)

    def scrape(self, video_urls, output_csv="data_tiktok.csv"):
        run_input = {
            "commentsPerPost": 500,
            "excludePinnedPosts": False,
            "maxRepliesPerComment": 10,
            "postURLs": video_urls,
            "resultsPerPage": 100
        }
        print("Menjalankan Apify Actor untuk scraping komentar TikTok...")
        run = self.client.actor("clockworks~tiktok-scraper").call(run_input=run_input)
        dataset_items = self.client.dataset(run["defaultDatasetId"]).list_items().items

        comments = []
        for idx, item in enumerate(dataset_items, 1):
            comments.append({
                "id": idx,
                "commentText": item.get("text", "")
            })

        df = pd.DataFrame(comments)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"Komentar disimpan di: {output_csv}")
        return df