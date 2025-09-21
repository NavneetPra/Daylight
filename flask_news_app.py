"""
Flask app that runs a daily scraper job and stays running to serve articles.

Fix: Previously the script terminated after running the scraper because of how the scheduler job was added.
Now we:
- Use BackgroundScheduler with daemon=False so it keeps the main process alive.
- Ensure Flask app.run() is blocking, keeping the server running.

Run with:
    python flask_news_app.py
and visit http://localhost:5000
"""

from flask import Flask, jsonify
import sqlite3, json
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import scraper_worker_topic_detection_scoring as sw

DB_PATH = getattr(sw, 'DB_PATH', 'news.db')
FAISS_INDEX_PATH = getattr(sw, 'FAISS_INDEX_PATH', 'faiss.index')
FEEDS = getattr(sw, 'FEEDS', [])

app = Flask(__name__)

sw.init_db(DB_PATH)
faiss_store = sw.FaissStore(sw.EMBED_DIM, path=FAISS_INDEX_PATH)
faiss_store.load_meta()

# --- Scraper job ---
def scrape_once():
    print(f"Starting scrape at {datetime.utcnow().isoformat()} UTC")
    conn = sqlite3.connect(DB_PATH)
    try:
        for feed in FEEDS:
            try:
                items = sw.fetch_feed_articles(feed)
                for e in items:
                    url = e.get('url')
                    if not url:
                        continue
                    sw.process_url(url, faiss_store, conn)
            except Exception as ex:
                print('feed error', feed, ex)
        faiss_store.save()
        print('Scrape finished — faiss saved')
    finally:
        conn.close()

# schedule daily scrape + initial scrape
scheduler = BackgroundScheduler(daemon=False)
scheduler.add_job(scrape_once, 'interval', hours=24, next_run_time=datetime.now())
scheduler.start()

@app.route('/api/articles')
def list_articles():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute('SELECT id, title, url, excerpt, score FROM articles ORDER BY score DESC LIMIT 20')
    rows = cur.fetchall()
    conn.close()
    results = []
    for r in rows:
        try:
            topics = json.loads(r['topics']) if r['topics'] else []
        except Exception:
            topics = []
        results.append({
            'id': r['id'],
            'title': r['title'],
            'url': r['url'],
            'excerpt': r['excerpt'],
            'topics': topics,
            'score': r['score'],
        })
    return jsonify(results)

@app.route('/')
def homepage():
    return """
    <!doctype html>
    <html><head><meta charset='utf-8'><title>News Demo</title></head>
    <body>
      <h1>Underreported News</h1>
      <div id='list'></div>
      <script>
        fetch('/api/articles').then(r=>r.json()).then(data=>{
          const list=document.getElementById('list');
          data.forEach(a=>{
            const div=document.createElement('div');
            div.innerHTML=`<p><b>${a.title}</b> - <a href="${a.url}" target="_blank">Read</a></p>`;
            list.appendChild(div);
          });
        });
      </script>
    </body></html>
    """

if __name__ == '__main__':
    # Flask server keeps running; scheduler runs in background
    app.run(host='0.0.0.0', port=5001, debug=True)