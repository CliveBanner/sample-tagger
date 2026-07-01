import sqlite3
import re
from collections import Counter
import math

def tokenize(path):
    parts = path.replace('\\', '/').split('/')
    important_parts = parts[-3:]
    tokens = []
    for p in important_parts:
        for t in re.split(r'[^a-zA-Z0-9]+', p.lower()):
            if len(t) > 2 and not t.isdigit() and t not in ('wav', 'mp3', 'aif', 'aiff', 'flac', 'ogg', 'the', 'and', 'for'):
                tokens.append(t)
    return tokens

def generate_cluster_names(db_path="samples.db"):
    con = sqlite3.connect(db_path)
    
    con.execute("CREATE TABLE IF NOT EXISTS clusters (id INTEGER PRIMARY KEY, name TEXT)")
    
    all_paths = con.execute("SELECT cluster_id, path FROM samples WHERE cluster_id IS NOT NULL").fetchall()
    
    cluster_docs = {}
    for cid, path in all_paths:
        if cid not in cluster_docs:
            cluster_docs[cid] = []
        cluster_docs[cid].extend(tokenize(path))
        
    num_docs = len(cluster_docs)
    df = Counter()
    for cid, tokens in cluster_docs.items():
        for t in set(tokens):
            df[t] += 1
            
    updates = []
    for cid, tokens in cluster_docs.items():
        if not tokens:
            updates.append((cid, ""))
            continue
            
        tfidf = {}
        tf = Counter(tokens)
        for t, freq in tf.items():
            idf = math.log(num_docs / (1 + df[t]))
            tfidf[t] = freq * idf
            
        top_tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:3]
        name = " ".join(t for t, score in top_tfidf).title()
        updates.append((cid, name))
        
    con.execute("BEGIN TRANSACTION")
    con.execute("DELETE FROM clusters")
    con.executemany("INSERT INTO clusters (id, name) VALUES (?, ?)", updates)
    con.commit()
    con.close()
    
    print(f"Generated names for {len(updates)} clusters.")

if __name__ == "__main__":
    generate_cluster_names()
