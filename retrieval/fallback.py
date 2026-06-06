from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf        = None
tfidf_matrix = None
grants_index = []

def build_tfidf_index(grants_list):
    global tfidf, tfidf_matrix, grants_index
    grants_index = grants_list
    texts        = [g['title'] + ' ' + g.get('guidelines_text', g.get('guidelines', ''))
                    for g in grants_list]
    tfidf        = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(texts)
    print(f"TF-IDF index built: {len(grants_list)} grants.")

def keyword_retrieve(researcher_text, top_k=10):
    query  = tfidf.transform([researcher_text])
    scores = cosine_similarity(query, tfidf_matrix).flatten()
    top_idx = scores.argsort()[-top_k:][::-1]
    return [(grants_index[i]['id'], round(float(scores[i]), 4))
            for i in top_idx]