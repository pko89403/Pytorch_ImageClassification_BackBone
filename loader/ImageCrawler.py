from icrawler.builtin import GoogleImageCrawler
from datetime import datetime
import os

def googleImageCrawler(feeder_threads = 1, parser_threads = 1, downloader_threads = 1, keyword=None):
        BASE_PATH = os.path.join( os.getcwd(), "dataset/raw")
        feeder_threads = feeder_threads
        parser_threads = parser_threads
        downloader_threads = downloader_threads
        keyword = keyword
        
        googleImgPath = os.path.join(BASE_PATH, keyword)
        download_path = {'root_dir' : googleImgPath}
        
        google_crawler = GoogleImageCrawler(feeder_threads = feeder_threads,
                                            parser_threads = parser_threads,
                                            downloader_threads = downloader_threads,
                                            storage = download_path)

        google_crawler.crawl(keyword = "cosmetic " + keyword, 
                             max_num = 1000)

if __name__ == "__main__":
    brands = ["sulwhasoo", "hera", "primera", "vitalbeauty", "iope", "laneige", "mamonde"]
    
    for brand in brands:
        googleImageCrawler(feeder_threads=4,
                            parser_threads=4,
                            downloader_threads=4,
                            keyword=brand)
