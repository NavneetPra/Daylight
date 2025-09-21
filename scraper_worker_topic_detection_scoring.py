"""
Scraper Worker with Topic Detection & Scoring
============================================

This file is a runnable prototype for a scraping worker that:
- Polls RSS feeds (feedparser)
- Extracts article content (newspaper3k)
- Generates embeddings (sentence-transformers)
- Builds topic centroids from short seed documents
- Detects relevant topics via cosine similarity
- Scores articles with a configurable ranking function
- Stores articles in a SQLite DB and embeddings in a FAISS index

Notes / Requirements
--------------------
- Python 3.10+
- pip install -r requirements.txt

requirements.txt (suggested):
    feedparser
    newspaper3k
    sentence-transformers
    faiss-cpu
    beautifulsoup4
    requests

Usage
-----
- Edit the FEEDS list and TOPIC_SEEDS as needed.
- Run: python scraper-worker-topic-detection-scoring.py
- The script will create `news.db` and `faiss.index` metadata files in the working directory.

This is a prototype — in production you'd add error handling, logging, rate-limiting, per-site rules, and an async worker queue.

"""

import os
import time
import json
import math
import sqlite3
import threading
from typing import List, Dict, Tuple
from datetime import datetime, timezone

import feedparser
import requests
from newspaper import Article
from sentence_transformers import SentenceTransformer, util
import faiss

# -----------------------------
# Configuration
# -----------------------------
DB_PATH = 'news.db'
FAISS_INDEX_PATH = 'faiss.index'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # 384-dim
EMBED_DIM = 384
SCRAPE_INTERVAL_SECONDS = 60 * 10  # 10 minutes (adjust)

# Example feeds - add more local/regional outlets and NGO sources
FEEDS = [
    'https://www.aljazeera.com/xml/rss/all.xml',
    'https://www.reutersagency.com/feed/?best-topics=world',
    # Add more feeds here, including smaller regional outlets
]

# Underreported topic seeds (short representative texts). Expand for better coverage.
TOPIC_SEEDS = {
    'Yemen': [
        'Yemen conflict humanitarian crisis Houthi government Saudi-led coalition civilian casualties famine blockade',
        'Aden Sanaa Taiz Hodeidah peace talks UN OCHA aid workers',
    ],
    'Syria': [
        'Syria civil war Aleppo Idlib Damascus regime opposition ISIS foreign intervention displacement refugees',
    ],
    'Sudan': [
        'Sudan conflict Darfur Khartoum Rapid Support Forces RSF humanitarian crisis coup',
    ],
    'DRC': [
        'Democratic Republic of Congo eastern DRC militia conflict Kivu Ebola displacement mining',
    ],
    'Myanmar': [
        'Myanmar coup military junta Rohingya conflict resistance junta displacement human rights',
    ],
}

# Topic boost (how strongly to boost underreported topics in final score)
TOPIC_BOOSTS = {k: 1.5 for k in TOPIC_SEEDS.keys()}  # 1.5 => 50% boost multiplier scaled by relevance

# Minimal source trust scores (manual), default 1.0
SOURCE_TRUST = {
    'Al Jazeera': 1.0,
    'Reuters': 1.0,
}

# -----------------------------
# Utilities: DB, FAISS
# -----------------------------

def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE,
        title TEXT,
        source TEXT,
        published_at TEXT,
        scraped_at TEXT,
        content TEXT,
        excerpt TEXT,
        topics TEXT, -- JSON array
        score REAL,
        embedding_id INTEGER
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS embeddings_map (
        embedding_id INTEGER PRIMARY KEY,
        article_id INTEGER,
        FOREIGN KEY(article_id) REFERENCES articles(id)
    )
    ''')
    conn.commit()
    conn.close()


class FaissStore:
    def __init__(self, dim: int, path: str = FAISS_INDEX_PATH):
        self.dim = dim
        self.path = path
        self.lock = threading.Lock()
        if os.path.exists(path):
            try:
                self.index = faiss.read_index(path)
            except Exception:
                print('Failed to read FAISS index - creating new')
                self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = faiss.IndexFlatIP(dim)
        # We'll maintain a simple python list mapping index positions to embedding_id
        self.id_map = []

    def save(self):
        with self.lock:
            faiss.write_index(self.index, self.path)
            with open(self.path + '.meta', 'w', encoding='utf-8') as f:
                json.dump(self.id_map, f)

    def load_meta(self):
        meta_path = self.path + '.meta'
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.id_map = json.load(f)

    def add(self, vec: List[float], embedding_id: int):
        with self.lock:
            import numpy as np
            v = np.array([vec], dtype='float32')
            # for cosine similarity using inner product, we must normalize vectors
            faiss.normalize_L2(v)
            self.index.add(v)
            self.id_map.append(embedding_id)

    def search(self, vec: List[float], top_k: int = 5) -> List[Tuple[int, float]]:
        import numpy as np
        v = np.array([vec], dtype='float32')
        faiss.normalize_L2(v)
        D, I = self.index.search(v, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0:
                continue
            emb_id = self.id_map[idx]
            results.append((emb_id, float(score)))
        return results


# -----------------------------
# Embeddings & Topic Centroids
# -----------------------------

print('Loading embedding model...')
EMB_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
print('Embedding model loaded.')


def compute_embedding(text: str):
    # returns normalized vector as python list
    emb = EMB_MODEL.encode(text, convert_to_numpy=True)
    # normalize
    import numpy as np
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype('float32')


def build_topic_centroids(topic_seeds: Dict[str, List[str]]):
    centroids = {}
    for topic, seeds in topic_seeds.items():
        vecs = []
        for s in seeds:
            vec = EMB_MODEL.encode(s, convert_to_numpy=True)
            vecs.append(vec)
        import numpy as np
        if len(vecs) > 0:
            centroid = np.mean(vecs, axis=0)
            # normalize centroid
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            centroids[topic] = centroid.astype('float32')
    return centroids


TOPIC_CENTROIDS = build_topic_centroids(TOPIC_SEEDS)
print('Built topic centroids for:', list(TOPIC_CENTROIDS.keys()))

# -----------------------------
# Scraping & extraction
# -----------------------------

USER_AGENT = 'UnderreportedNewsAggregatorBot/0.1 (+https://example.org)'


def extract_article_text(url: str) -> Tuple[str, str, str]:
    """Returns (title, text, publish_date_iso or None)"""
    try:
        a = Article(url)
        a.download()
        a.parse()
        title = a.title or ''
        text = a.text or ''
        publish_date = None
        if a.publish_date:
            publish_date = a.publish_date.astimezone(timezone.utc).isoformat()
        return title, text, publish_date
    except Exception as e:
        print('extract error', e, url)
        return '', '', None


def fetch_feed_articles(feed_url: str) -> List[Dict]:
    parsed = feedparser.parse(feed_url)
    items = []
    for entry in parsed.entries:
        url = entry.get('link')
        if not url:
            continue
        items.append({'url': url, 'title': entry.get('title', ''), 'published': entry.get('published', None)})
    return items


# -----------------------------
# Topic detection & scoring
# -----------------------------


def detect_topics_for_text(text: str, top_k: int = 3) -> List[Tuple[str, float]]:
    # compute embedding for the text then measure cosine with topic centroids
    emb = EMB_MODEL.encode(text, convert_to_numpy=True)
    import numpy as np
    if np.linalg.norm(emb) > 0:
        emb = emb / np.linalg.norm(emb)
    scores = []
    for topic, centroid in TOPIC_CENTROIDS.items():
        score = float(np.dot(emb, centroid))  # cosine
        scores.append((topic, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def recency_score(published_iso: str) -> float:
    # half-life style decay (hours)
    try:
        if not published_iso:
            return 0.5
        pub = datetime.fromisoformat(published_iso.replace('Z', '+00:00'))
        delta = datetime.now(timezone.utc) - pub
        hours = delta.total_seconds() / 3600.0
        half_life = 24.0 * 3.0  # 3 days
        score = math.exp(-math.log(2) * hours / half_life)
        return score
    except Exception:
        return 0.5


def compute_final_score(base_relevance: float, topic_matches: List[Tuple[str, float]], source_name: str, published_iso: str) -> float:
    r_score = recency_score(published_iso)
    source_score = SOURCE_TRUST.get(source_name, 1.0)
    # topic_multiplier scales with the best topic match times its configured boost
    top_topic, top_score = topic_matches[0]
    boost = TOPIC_BOOSTS.get(top_topic, 1.0)
    topic_multiplier = 1.0 + (boost - 1.0) * (max(0.0, top_score))
    final = base_relevance * r_score * source_score * topic_multiplier
    return float(final)


# -----------------------------
# Ingest & deduplication
# -----------------------------


def article_exists(conn: sqlite3.Connection, url: str) -> bool:
    cur = conn.cursor()
    cur.execute('SELECT 1 FROM articles WHERE url = ?', (url,))
    return cur.fetchone() is not None


def save_article(conn: sqlite3.Connection, item: Dict, embedding_id: int):
    cur = conn.cursor()
    cur.execute('''
        INSERT OR IGNORE INTO articles (url, title, source, published_at, scraped_at, content, excerpt, topics, score, embedding_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        item['url'],
        item['title'],
        item['source'],
        item['published_at'],
        item['scraped_at'],
        item['content'],
        item['excerpt'],
        json.dumps(item['topics']),
        item['score'],
        embedding_id,
    ))
    conn.commit()
    cur.execute('SELECT id FROM articles WHERE url = ?', (item['url'],))
    row = cur.fetchone()
    article_id = row[0] if row else None
    if article_id:
        # record embedding mapping
        cur.execute('INSERT OR IGNORE INTO embeddings_map (embedding_id, article_id) VALUES (?, ?)', (embedding_id, article_id))
        conn.commit()
    return article_id


# -----------------------------
# Orchestration: process a single URL
# -----------------------------


def process_url(url: str, faiss_store: FaissStore, conn: sqlite3.Connection):
    if article_exists(conn, url):
        print('Already have', url)
        return None

    title, text, published = extract_article_text(url)
    if not text or len(text.split()) < 50:
        print('Skipping short/empty article', url)
        return None

    # basic excerpt
    excerpt = ' '.join(text.split()[:50]) + '...'

    # topic detection
    topic_scores = detect_topics_for_text(text, top_k=3)
    topics = [{'name': t, 'score': s} for t, s in topic_scores if s > 0.25]

    # base relevance: use embedding similarity to the top topic centroid as a proxy
    top_rel = topic_scores[0][1]
    base_relevance = max(0.01, (top_rel + 1.0) / 2.0)  # map cosine [-1,1] -> [0,1]

    source_name = ''
    try:
        # naive source extraction from URL hostname
        from urllib.parse import urlparse
        parsed = urlparse(url)
        source_name = parsed.hostname or ''
    except Exception:
        source_name = ''

    scraped_at = datetime.now(timezone.utc).isoformat()

    final_score = compute_final_score(base_relevance, topic_scores, source_name, published)

    # embedding for storage
    emb = compute_embedding(text)

    # create a new embedding id (simple counter)
    embedding_id = len(faiss_store.id_map) + 1
    faiss_store.add(emb.tolist(), embedding_id)

    item = {
        'url': url,
        'title': title,
        'source': source_name,
        'published_at': published,
        'scraped_at': scraped_at,
        'content': text,
        'excerpt': excerpt,
        'topics': topics,
        'score': final_score,
    }

    article_id = save_article(conn, item, embedding_id)
    print('Saved', url, 'as article_id', article_id, 'score', final_score, 'topics', topics)
    return article_id


# -----------------------------
# Main loop: poll feeds
# -----------------------------


def main_loop():
    init_db()
    faiss_store = FaissStore(EMBED_DIM)
    faiss_store.load_meta()

    conn = sqlite3.connect(DB_PATH)

    while True:
        for feed in FEEDS:
            try:
                print('Polling feed', feed)
                items = fetch_feed_articles(feed)
                for e in items:
                    url = e['url']
                    try:
                        process_url(url, faiss_store, conn)
                    except Exception as ex:
                        print('error processing url', url, ex)
            except Exception as ex:
                print('feed error', feed, ex)
        # persist faiss meta + index
        faiss_store.save()
        print('Sleeping for', SCRAPE_INTERVAL_SECONDS, 'seconds')
        time.sleep(SCRAPE_INTERVAL_SECONDS)


if __name__ == '__main__':
    main_loop()