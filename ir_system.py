import json
import os
import string
import math
import sys
from collections import defaultdict
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

# (If you have not already downloaded these NLTK resources, you may need to run these lines once.)
# nltk.download('stopwords')
# nltk.download('punkt')

# ------------------------------------------------------------------------------
# Step 1: Preprocessing Functions
# ------------------------------------------------------------------------------

def tokenize_and_remove_punctuations(text):
    """
    Remove punctuation and digits from text, then tokenize.
    """
    translator = str.maketrans('', '', string.punctuation)
    text_no_punct = text.translate(translator)
    text_no_digits = ''.join(ch for ch in text_no_punct if not ch.isdigit())
    tokens = wordpunct_tokenize(text_no_digits.lower())
    return tokens

def get_stopwords():
    """
    Return a set of English stopwords from NLTK.
    """
    return set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    """
    Tokenize, remove stopwords, filter out very short tokens,
    and stem the remaining tokens.
    """
    tokens = tokenize_and_remove_punctuations(text)
    stopwords = get_stopwords()
    filtered_tokens = [token for token in tokens if token not in stopwords and len(token) > 2]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens

# ------------------------------------------------------------------------------
# Step 2: Reading the Corpus, Queries, and Building the Inverted Index
# ------------------------------------------------------------------------------

def read_corpus(corpus_file):
    """
    Reads the corpus from a JSONL file where each line is a document.
    Each document should have an '_id', 'title', and 'text'.
    The title and text are concatenated and preprocessed.
    Returns a dictionary mapping doc_id to list of tokens.
    """
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc['_id']
            content = doc.get('title', '') + ' ' + doc.get('text', '')
            tokens = preprocess_text(content)
            corpus[doc_id] = tokens
    return corpus

def get_relevance(relevance_file):
    """
    Reads the relevance judgments from a TSV file.
    Each line is expected to have: query_id, unused_field, doc_id, relevance
    Only documents with a relevance greater than zero are considered relevant.
    Returns a dictionary mapping query_id to a list of relevant doc_ids.
    """
    relevances = defaultdict(list)
    with open(relevance_file, 'r', encoding='utf-8') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split('\t')
            if int(rel) > 0:
                relevances[qid].append(docid)
    return relevances

def read_queries(queries_file, valid_query_ids):
    """
    Reads the queries from a JSONL file.
    Only queries with IDs that are in the valid_query_ids set (derived from the relevance file)
    are processed.
    Returns a dictionary mapping query_id to a list of preprocessed tokens.
    """
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            q_id = q['_id']
            if q_id in valid_query_ids:
                tokens = preprocess_text(q.get('text', ''))
                queries[q_id] = tokens
    return queries

def build_inverted_index(corpus):
    """
    Build an inverted index mapping each term to a set of document IDs in which it occurs.
    """
    inverted_index = defaultdict(set)
    for doc_id, tokens in corpus.items():
        for token in set(tokens):
            inverted_index[token].add(doc_id)
    return inverted_index

# ------------------------------------------------------------------------------
# Step 3: TF-IDF, Cosine Similarity, and Retrieval
# ------------------------------------------------------------------------------

def calculate_tf(tokens):
    """
    Calculate term frequency (TF) for a list of tokens.
    """
    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    return tf

def calculate_idf(corpus):
    """
    Calculate inverse document frequency (IDF) for each term in the corpus.
    """
    idf = {}
    N = len(corpus)
    term_doc_count = defaultdict(int)
    for tokens in corpus.values():
        for term in set(tokens):
            term_doc_count[term] += 1
    for term, df in term_doc_count.items():
        idf[term] = math.log(N / df) if df > 0 else 0
    return idf

def calculate_tfidf(tf, idf):
    """
    Compute the TF-IDF weight for each term given its TF and the precomputed IDF.
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

# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------

def main():
    if len(sys.argv) != 4:
        print("Usage: python ir_system.py corpus.jsonl queries.jsonl test.tsv")
        sys.exit(1)
    
    corpus_file = sys.argv[1]
    queries_file = sys.argv[2]
    relevance_file = sys.argv[3]
    
    # Read the relevance judgments first so we know which queries to process.
    print("Reading relevance judgments...")
    relevances = get_relevance(relevance_file)
    valid_query_ids = set(relevances.keys())
    print(f"Number of queries in relevance file: {len(valid_query_ids)}")
    
    # Read and preprocess the corpus and queries.
    print("Reading and preprocessing corpus...")
    corpus = read_corpus(corpus_file)
    print(f"Number of documents in corpus: {len(corpus)}")
    
    print("Reading and preprocessing queries...")
    queries = read_queries(queries_file, valid_query_ids)
    print(f"Number of queries to process: {len(queries)}")
    
    # Build the inverted index for fast access during retrieval.
    print("Building inverted index...")
    inverted_index = build_inverted_index(corpus)
    
    # Calculate IDF across the entire corpus.
    print("Calculating IDF...")
    idf = calculate_idf(corpus)
    
    # Precompute the TF-IDF vector for each document.
    print("Computing TF-IDF for each document...")
    doc_tfidf = {}
    for doc_id, tokens in corpus.items():
        tf = calculate_tf(tokens)
        doc_tfidf[doc_id] = calculate_tfidf(tf, idf)
    
    # Dictionaries to store top 100 results per query.
    top100_results = {}  # For each query, only the top 100 candidate docs
    
    print("Processing queries and retrieving documents...")
    # Process each query
    for query_id in sorted(queries, key=lambda x: int(x)):
        query_tokens = queries[query_id]
        query_tf = calculate_tf(query_tokens)
        query_tfidf = calculate_tfidf(query_tf, idf)
        
        # Retrieve candidate documents (those containing at least one query term)
        candidate_docs = set()
        for token in query_tokens:
            candidate_docs.update(inverted_index.get(token, set()))
        
        # Compute cosine similarity for each candidate document.
        scores = {}
        for doc_id in candidate_docs:
            score = cosine_similarity(query_tfidf, doc_tfidf[doc_id])
            scores[doc_id] = score
        
        # Rank all candidate documents by descending score and store only the top 100.
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top100_results[query_id] = sorted_docs[:100]
    
    # Write the top 100 ranking results to Results.txt in TREC format.
    print("Writing top 100 rankings to Results.txt...")
    with open("Results.txt", "w", encoding="utf-8") as out_file:
        for query_id in sorted(top100_results, key=lambda x: int(x)):
            rank = 1
            for doc_id, score in top100_results[query_id]:
                out_file.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} myIRsystem\n")
                rank += 1
    
    # Optionally, print the top 100 results to the console.
    # print("\nTop 100 Results for Each Query:")
    # for query_id in sorted(top100_results, key=lambda x: int(x)):
    #     print(f"\nQuery {query_id}:")
    #     for rank, (doc_id, score) in enumerate(top100_results[query_id], 1):
    #         print(f"{query_id} Q0 {doc_id} {rank} {score:.4f} myIRsystem")
    
    print("\nRetrieval complete. Top 100 results for each query are saved in Results.txt.")


if __name__ == "__main__":
    main()
