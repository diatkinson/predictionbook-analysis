import argparse
import os
import time

import requests
from lxml import html
from tqdm import tqdm

import pba.parse as parse


def scrape_page(page_dir: str, page_id: int) -> None:
    page_url = "https://predictionbook.com/predictions/" + str(page_id)
    start_time = time.time()
    resp = requests.get(page_url)
    resp_time = time.time() - start_time
    fname = os.path.join(page_dir, f"page-{page_id}.html")
    with open(fname, "w") as f:
        f.write(resp.text)
    delay = 5 * resp_time
    time.sleep(delay)


def scrape(args: argparse.Namespace) -> None:
    """
    Args:
      --page-dir (required): place to store scraped pages / look for pages we've downloaded already
      --from-scratch: if present, scrape everything we can get, regardless of the page is useful

    Scrape all the new or reasonably-suspected-to-be-updated pages. Or, with --from-scratch, all
    the pages.
    """
    predictions_page = html.fromstring(requests.get("https://predictionbook.com/predictions").text)
    last_public_page = max(int(url.split('/')[-1])
                           for url in predictions_page.xpath("//span[@class='title']/a/@href"))
    # Scrape everything
    if args.from_scratch:
        print(f"Scraping all pages, from 1 to {last_public_page}")
        for page_id in tqdm(range(1, last_public_page+1)):
            scrape_page(args.page_dir, page_id)
    else:
        scraped_page_ids = [int(page.split(".")[0].split("-")[1])
                            for page in os.listdir(args.page_dir) if page.endswith(".html")]
        last_scraped_page = max(scraped_page_ids) if scraped_page_ids else 0
        print(f"Updating {last_scraped_page} pages, ignoring private and resolved pages.")
        for page_id in tqdm(range(1, last_scraped_page+1)):
            page_file = os.path.join(args.page_dir, f"page-{page_id}.html")
            is_file = os.path.exists(page_file)
            prediction = parse.parse_file(page_file)
            if is_file and (prediction is not None) and prediction.unknown():
                scrape_page(args.page_dir, page_id)
        print(f"Scraping new pages from page {last_scraped_page+1} through page {last_public_page}")
        for page_id in tqdm(range(last_scraped_page+1, last_public_page+1)):
            scrape_page(args.page_dir, page_id)
