import sqlite3
import os
import re
from collections import Counter
import math

DB = "samples.db"

def tokenize(path):
    # Extract filename and the last few directory names
    parts = path.replace('\\', '/').split('/')
    
    # Focus on the last 3 parts (e.g. parent_dir/sub_dir/filename.wav)
    # as they contain the most descriptive information.
    important_parts = parts[-3:]
    
    # Split by non-alphanumeric, convert to lowercase
    tokens = []
    for p in important_parts:
        for t in re.split(r'[^a-zA-Z0-9]+', p.lower()):
            # Filter out very short tokens, pure numbers, and extensions
            if len(t) > 2 and not t.isdigit() and t not in ('wav', 'mp3', 'aif', 'aiff', 'flac', 'ogg'):
                tokens.append(t)
    return tokens

def main():
    con = sqlite3.connect(DB)
    
    # Get top 20 clusters by size
    top_clusters = con.execute("""
        SELECT cluster_id, COUNT(*) as c 
        FROM samples 
        WHERE cluster_id IS NOT NULL 
        GROUP BY cluster_id 
        ORDER BY c DESC 
        LIMIT 20
    """).fetchall()
    
    print("Parsing paths and building Document Frequencies...")
    all_paths = con.execute("SELECT cluster_id, path FROM samples WHERE cluster_id IS NOT NULL").fetchall()
    
    # Treat each cluster as a single "document" of tokens
    cluster_docs = {}
    for cid, path in all_paths:
        if cid not in cluster_docs:
            cluster_docs[cid] = []
        cluster_docs[cid].extend(tokenize(path))
        
    num_docs = len(cluster_docs)
    df = Counter()
    for cid, tokens in cluster_docs.items():
        # Count each token once per document (cluster)
        for t in set(tokens):
            df[t] += 1
            
    print(f"\n--- Testing Naming Strategies on Top {len(top_clusters)} Clusters ---\n")
    
    for cid, count in top_clusters:
        tokens = cluster_docs[cid]
        
        # Method 1: Most Common Tokens (Simple Frequency)
        tc = Counter(tokens)
        most_common = [t for t, c in tc.most_common(4)]
        
        # Method 2: TF-IDF
        tfidf = {}
        tf = Counter(tokens)
        for t, freq in tf.items():
            # Term frequency * Inverse Document Frequency
            # We add 1 to df[t] to prevent division by zero just in case
            idf = math.log(num_docs / (1 + df[t]))
            tfidf[t] = freq * idf
            
        top_tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:4]
        tfidf_words = [t for t, score in top_tfidf]
        
        print(f"Cluster {cid} ({count} samples):")
        print(f"  M1 (Frequency) : {' '.join(most_common).title()}")
        print(f"  M2 (TF-IDF)    : {' '.join(tfidf_words).title()}")
        
        # Show a random path for context
        sample_path = con.execute("SELECT path FROM samples WHERE cluster_id=? LIMIT 1", (cid,)).fetchone()[0]
        print(f"  Sample path    : {os.path.basename(os.path.dirname(sample_path))}/{os.path.basename(sample_path)}")
        print()

if __name__ == "__main__":
    main()
