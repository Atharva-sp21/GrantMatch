from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

embedder = SentenceTransformer('allenai-specter')
client   = QdrantClient(':memory:')   # swap to 'localhost' in production

COLLECTION = 'grant_guidelines'
THRESHOLD  = 0.45

def setup_collection():
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    print(f"Collection '{COLLECTION}' created.")

def embed_grants(grants_list):
    points = []
    for i, grant in enumerate(grants_list):
        text   = grant['title'] + ' ' + grant.get('guidelines_text', grant.get('guidelines', ''))
        vector = embedder.encode(text).tolist()
        points.append(PointStruct(
            id=i,
            vector=vector,
            payload={
                'grant_id':  grant['id'],
                'agency':    grant.get('agency', ''),
                'deadline':  grant.get('deadline', ''),
                'amount':    grant.get('funding_amount', 0),
                'title':     grant['title'],
            }
        ))
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Stored {len(points)} grant embeddings.")

def get_researcher_text(researcher):
    abstracts = researcher.get('recent_abstracts', [])
    return ' [SEP] '.join(abstracts[:5])

def rag_retrieve(researcher, top_k=10):
    query_text   = get_researcher_text(researcher)
    query_vector = embedder.encode(query_text).tolist()
    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )
    return [(r.payload['grant_id'], round(r.score, 4)) for r in results]

def rag_score_single(researcher, grant_text):
    r_vec = embedder.encode(get_researcher_text(researcher))
    g_vec = embedder.encode(grant_text)
    # cosine similarity
    cos   = float(
        (r_vec @ g_vec) /
        (((r_vec**2).sum()**0.5) * ((g_vec**2).sum()**0.5))
    )
    return max(0.0, cos)