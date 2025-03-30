import json
import os
import string
import math
import sys
import argparse
from collections import defaultdict
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

# If you haven't downloaded NLTK resources, uncomment the lines below and run once:
# nltk.download('stopwords')
# nltk.download('punkt')

# -------------------------------------------------------------------------------
# Step 1: Preprocessing Functions
# -------------------------------------------------------------------------------

def tokenize_and_remove_punctuations(text):
    """
    Remove punctuation and digits, then tokenize.
    """
    translator = str.maketrans('', '', string.punctuation)
    text_no_punct = text.translate(translator)
    text_no_digits = ''.join(ch for ch in text_no_punct if not ch.isdigit())
    tokens = wordpunct_tokenize(text_no_digits.lower())
    return tokens

def get_stopwords():
    """
    Return the set of English stopwords from NLTK.
    """
    return set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    """
    Tokenize, remove stopwords, filter out words with length < 3, then apply stemming.
    Return the list of processed tokens.
    """
    tokens = tokenize_and_remove_punctuations(text)
    stopwords = get_stopwords()
    filtered_tokens = [token for token in tokens if token not in stopwords and len(token) > 2]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens

# -------------------------------------------------------------------------------
# Step 2: Data Reading – Corpus, Queries, and Relevance Annotations
# To support neural retrieval, original text is preserved during loading
# -------------------------------------------------------------------------------

def read_corpus_with_raw(corpus_file):
    """
    Read the corpus.jsonl file, each line is a document with '_id', 'title', and 'text'.
    Concatenate title and text, perform preprocessing to generate tokens,
    and retain the original text (raw) for Dense retrieval or BERT re-ranking.
    Returns a dict: doc_id -> {"tokens": [...], "raw": full text}
    """
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc['_id']
            content = doc.get('title', '') + ' ' + doc.get('text', '')
            tokens = preprocess_text(content)
            corpus[doc_id] = {"tokens": tokens, "raw": content}
    return corpus

def get_relevance(relevance_file):
    """
    Read relevance annotation file (TSV format), format: query_id, unused_field, doc_id, relevance
    Only relevance > 0 is considered relevant.
    Returns dict: query_id -> [doc_id, ...]
    """
    relevances = defaultdict(list)
    with open(relevance_file, 'r', encoding='utf-8') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split('\t')
            if int(rel) > 0:
                relevances[qid].append(docid)
    return relevances

def read_queries_with_raw(queries_file, valid_query_ids):
    """
    Read queries.jsonl file, one query per line.
    Only process queries in valid_query_ids set.
    For each query, generate tokens via preprocessing and also keep the original text (raw).
    Returns dict: query_id -> {"tokens": [...], "raw": query text}
    """
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            q_id = q['_id']
            if q_id in valid_query_ids:
                raw_text = q.get('text', '')
                tokens = preprocess_text(raw_text)
                queries[q_id] = {"tokens": tokens, "raw": raw_text}
    return queries

def build_inverted_index(corpus):
    """
    Build an inverted index: for each document's tokens (stored in corpus[doc_id]["tokens"]), create a mapping.
    Returns dict: term -> set(doc_id)
    """
    inverted_index = defaultdict(set)
    for doc_id, data in corpus.items():
        tokens = data["tokens"]
        for token in set(tokens):
            inverted_index[token].add(doc_id)
    return inverted_index

# -------------------------------------------------------------------------------
# Step 3: Traditional TF-IDF Retrieval (Baseline Retrieval)
# -------------------------------------------------------------------------------

def calculate_tf(tokens):
    """
    Calculate term frequency (TF) from tokens.
    """
    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    return tf

def calculate_idf(corpus):
    """
    Compute inverse document frequency (IDF) for the entire corpus.
    Corpus is a dict where tokens are in data["tokens"].
    """
    idf = {}
    N = len(corpus)
    term_doc_count = defaultdict(int)
    for data in corpus.values():
        for term in set(data["tokens"]):
            term_doc_count[term] += 1
    for term, df in term_doc_count.items():
        idf[term] = math.log(N / df) if df > 0 else 0
    return idf

def calculate_tfidf(tf, idf):
    """
    Calculate TF-IDF weights using TF and precomputed IDF.
    """
    tfidf = {}
    for term, freq in tf.items():
        tfidf[term] = freq * idf.get(term, 0)
    return tfidf

def cosine_similarity(query_vec, doc_vec):
    """
    Compute cosine similarity between two vectors represented as dictionaries.
    """
    dot_product = 0.0
    for term, weight in query_vec.items():
        if term in doc_vec:
            dot_product += weight * doc_vec[term]
    query_norm = math.sqrt(sum(weight**2 for weight in query_vec.values()))
    doc_norm = math.sqrt(sum(weight**2 for weight in doc_vec.values()))
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return dot_product / (query_norm * doc_norm)

def baseline_retrieval(queries, corpus, inverted_index, idf, top_k=100):
    """
    Retrieve using TF-IDF and cosine similarity:
      1. Compute TF-IDF vector for each query (tokens are in queries[query_id]["tokens"]).
      2. Use inverted index to get candidate documents that contain at least one query term.
      3. Compute TF-IDF vectors for candidate documents (tokens from corpus[doc_id]["tokens"]), and compute similarity.
      4. Return top_k results for each query as a list [(doc_id, score), ...].
    """
    results = {}
    for query_id in sorted(queries, key=lambda x: int(x)):
        query_tokens = queries[query_id]["tokens"]
        query_tf = calculate_tf(query_tokens)
        query_tfidf = calculate_tfidf(query_tf, idf)
        
        candidate_docs = set()
        for token in query_tokens:
            candidate_docs.update(inverted_index.get(token, set()))
        
        scores = {}
        for doc_id in candidate_docs:
            doc_tf = calculate_tf(corpus[doc_id]["tokens"])
            doc_tfidf = calculate_tfidf(doc_tf, idf)
            score = cosine_similarity(query_tfidf, doc_tfidf)
            scores[doc_id] = score
        
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        results[query_id] = sorted_docs[:top_k]
    return results

# -------------------------------------------------------------------------------
# Step 4: Dense Retrieval (using Sentence-BERT)
# -------------------------------------------------------------------------------

def dense_retrieval(queries, corpus, top_k=100, model_name="all-MiniLM-L6-v2"):
    """
    Use pretrained SentenceTransformer model to convert documents and queries into dense vectors,
    then retrieve based on cosine similarity.
    Returns: dict query_id -> [(doc_id, score), ...]
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print("Loading dense model:", model_name)
    model = SentenceTransformer(model_name)
    
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[doc_id]["raw"] for doc_id in doc_ids]
    print("Computing document embeddings...")
    doc_embeddings = model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=True)
    
    results = {}
    for query_id in sorted(queries, key=lambda x: int(x)):
        query_text = queries[query_id]["raw"]
        query_embedding = model.encode(query_text, convert_to_numpy=True)
        
        doc_norms = np.linalg.norm(doc_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            cosine_scores = np.zeros(len(doc_embeddings))
        else:
            cosine_scores = np.dot(doc_embeddings, query_embedding) / (doc_norms * query_norm)
        
        top_indices = np.argsort(-cosine_scores)[:top_k]
        top_candidates = [(doc_ids[i], float(cosine_scores[i])) for i in top_indices]
        results[query_id] = top_candidates
    return results

# -------------------------------------------------------------------------------
# Step 5: BERT Re-Ranking (using Cross-Encoder)
# -------------------------------------------------------------------------------

def bert_rerank(queries, corpus, baseline_results, top_k=100, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    For each query, use baseline_results (e.g., TF-IDF candidates) as input,
    then construct [query, doc_text] pairs for the CrossEncoder to score relevance,
    finally re-rank candidates by scores and return top_k results.
    """
    from sentence_transformers import CrossEncoder
    print("Loading cross-encoder model:", model_name)
    model = CrossEncoder(model_name)
    
    results = {}
    for query_id in sorted(baseline_results, key=lambda x: int(x)):
        query_text = queries[query_id]["raw"]
        candidates = baseline_results[query_id]
        candidate_pairs = []
        candidate_doc_ids = []
        for doc_id, _ in candidates:
            doc_text = corpus[doc_id]["raw"]
            candidate_pairs.append([query_text, doc_text])
            candidate_doc_ids.append(doc_id)
        
        print(f"Re-ranking query {query_id} with {len(candidate_pairs)} candidates...")
        scores = model.predict(candidate_pairs)
        candidate_score_pairs = list(zip(candidate_doc_ids, scores))
        candidate_score_pairs.sort(key=lambda x: x[1], reverse=True)
        results[query_id] = candidate_score_pairs[:top_k]
    return results

# -------------------------------------------------------------------------------
# Step 6: Output Results (TREC Format)
# -------------------------------------------------------------------------------

def write_results(results, output_filename, run_tag):
    """
    Write top_k results for each query to the output file in the format:
    query_id Q0 doc_id rank score tag
    """
    with open(output_filename, "w", encoding="utf-8") as out_file:
        for query_id in sorted(results, key=lambda x: int(x)):
            rank = 1
            for doc_id, score in results[query_id]:
                out_file.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_tag}\n")
                rank += 1
    print(f"Results written to {output_filename}")

# -------------------------------------------------------------------------------
# Main function: Choose retrieval method based on command-line arguments
# -------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Assignment2: Neural Retrieval Methods")
    parser.add_argument("corpus_file", help="Path to corpus.jsonl")
    parser.add_argument("queries_file", help="Path to queries.jsonl")
    parser.add_argument("relevance_file", help="Path to test.tsv (relevance judgments)")
    parser.add_argument("--method", choices=["tfidf", "dense", "bert"], default="tfidf",
                        help="Choose retrieval method: 'tfidf' for traditional baseline, 'dense' for dense retrieval, 'bert' for BERT re-ranking")
    parser.add_argument("--dense_model", default="all-MiniLM-L6-v2", help="Model name for SentenceTransformer (Dense Retrieval)")
    parser.add_argument("--bert_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Model name for CrossEncoder (BERT Re-ranking)")
    parser.add_argument("--top_k", type=int, default=100, help="Number of documents to return per query")
    args = parser.parse_args()
    
    print("Reading relevance judgments...")
    relevances = get_relevance(args.relevance_file)
    valid_query_ids = set(relevances.keys())
    print(f"{len(valid_query_ids)} queries found in relevance file.")
    
    print("Reading and preprocessing corpus...")
    corpus = read_corpus_with_raw(args.corpus_file)
    print(f"Corpus contains {len(corpus)} documents.")
    
    print("Reading and preprocessing queries...")
    queries = read_queries_with_raw(args.queries_file, valid_query_ids)
    print(f"{len(queries)} queries to process.")
    
    run_tag = "myIRsystem"
    
    if args.method == "tfidf":
        print("Building inverted index...")
        inverted_index = build_inverted_index(corpus)
        print("Calculating IDF...")
        idf = calculate_idf(corpus)
        print("Running baseline TF-IDF retrieval...")
        results = baseline_retrieval(queries, corpus, inverted_index, idf, top_k=args.top_k)
        output_file = "Results_tfidf.txt"
    
    elif args.method == "dense":
        print("Running Dense Retrieval (Sentence-BERT)...")
        results = dense_retrieval(queries, corpus, top_k=args.top_k, model_name=args.dense_model)
        output_file = "Results_dense.txt"
    
    elif args.method == "bert":
        print("Building inverted index for candidate generation...")
        inverted_index = build_inverted_index(corpus)
        print("Calculating IDF...")
        idf = calculate_idf(corpus)
        print("Running baseline TF-IDF retrieval to get candidate docs...")
        baseline_results = baseline_retrieval(queries, corpus, inverted_index, idf, top_k=args.top_k)
        print("Re-ranking candidates using CrossEncoder...")
        results = bert_rerank(queries, corpus, baseline_results, top_k=args.top_k, model_name=args.bert_model)
        output_file = "Results_bert.txt"
    
    else:
        print("Invalid retrieval method selected.")
        sys.exit(1)
    
    write_results(results, output_file, run_tag)
    print("Retrieval completed.")

if __name__ == "__main__":
    main()
