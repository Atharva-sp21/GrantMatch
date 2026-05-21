from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

client   = QdrantClient(':memory:')

COLLECTION = 'grant_guidelines'
THRESHOLD  = 0.3

def setup_collection():
    try:
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=100, distance=Distance.COSINE)
        )
        print(f"[OK] Collection '{COLLECTION}' created.")
    except Exception as e:
        print(f"[WARN] Collection setup: {e}")

def embed_grants(grants_list):
    try:
        tfidf_vec = TfidfVectorizer(max_features=100, stop_words='english')
        texts = [g.get('title', '') for g in grants_list]
        tfidf_matrix = tfidf_vec.fit_transform(texts).toarray()

        points = []
        for i, grant in enumerate(grants_list):
            vector = tfidf_matrix[i].tolist()
            # Pad to 100 if needed
            while len(vector) < 100:
                vector.append(0.0)
            vector = vector[:100]

            points.append(PointStruct(
                id=i,
                vector=vector,
                payload={
                    'grant_id':  grant['id'],
                    'agency_id': grant.get('agency_id', 0),
                    'amount':    grant.get('amount', 0),
                    'title':     grant['title'],
                }
            ))
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"[OK] Stored {len(points)} grant embeddings.")
    except Exception as e:
        print(f"[WARN] Embed grants error: {e}")

def get_researcher_text(researcher):
    return researcher.get('name', '') + ' ' + str(researcher.get('h_index', 0))

def rag_retrieve(researcher, top_k=10):
    try:
        query_text = get_researcher_text(researcher)
        tfidf_vec = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_vec.fit([query_text])
        query_vector = tfidf_vec.transform([query_text]).toarray()[0].tolist()
        while len(query_vector) < 100:
            query_vector.append(0.0)
        query_vector = query_vector[:100]

        results = client.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        return [(r.payload['grant_id'], round(r.score, 4)) for r in results]
    except Exception as e:
        print(f"[WARN] RAG retrieve error: {e}")
        return []

def rag_score_single(researcher, grant_text):
    try:
        r_text = get_researcher_text(researcher)
        tfidf_vec = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_vec.fit([r_text, grant_text])
        r_vec = tfidf_vec.transform([r_text]).toarray()[0]
        g_vec = tfidf_vec.transform([grant_text]).toarray()[0]

        r_norm = np.linalg.norm(r_vec)
        g_norm = np.linalg.norm(g_vec)
        if r_norm == 0 or g_norm == 0:
            return 0.5

        cos = float((r_vec @ g_vec) / (r_norm * g_norm))
        return max(0.0, min(1.0, cos))
    except Exception:
        return 0.5