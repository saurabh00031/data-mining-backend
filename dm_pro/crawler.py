# webcrawlerapi/crawler.py
import requests
from bs4 import BeautifulSoup

def crawl_dfs(seed_url, max_depth):
    stack = [{'url': seed_url, 'depth': 0}]
    visited = set()
    crawled_urls = []

    while stack:
        current_page = stack.pop()
        url = current_page['url']
        depth = current_page['depth']

        if depth > max_depth or url in visited:
            continue

        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            print(f'Depth {depth}: {url}')

            visited.add(url)
            crawled_urls.append(url)

            links = [link.get('href') for link in soup.find_all('a') if link.get('href') and link.get('href') not in visited]
            stack.extend({'url': link, 'depth': depth + 1} for link in reversed(links))

        except requests.RequestException as error:
            print(f'Error crawling {url}: {error}')

    return crawled_urls

def crawl_bfs(seed_url, max_depth):
    queue = [{'url': seed_url, 'depth': 0}]
    visited = set()
    crawled_urls = []

    while queue:
        current_page = queue.pop(0)
        url = current_page['url']
        depth = current_page['depth']

        if depth > max_depth or url in visited:
            continue

        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            print(f'Depth {depth}: {url}')

            visited.add(url)
            crawled_urls.append(url)

            links = [link.get('href') for link in soup.find_all('a') if link.get('href') and link.get('href') not in visited]
            queue.extend({'url': link, 'depth': depth + 1} for link in links)

        except requests.RequestException as error:
            print(f'Error crawling {url}: {error}')

    return crawled_urls
