# **CSI 4107 Assignment 2 – Neural Information Retrieval System**


## Team Members and Task Distribution

Bruno Kazadi (300210848):
–Integrated the neural retrieval modules. Assisted in debugging and performance evaluation of the IR system.

Bernardo Caiado (300130165):
– Contributed to data preprocessing and candidate generation via the traditional TF-IDF module.

Jun Ning (300286811):
– Integrated the neural retrieval modules， designed and implemented the re-ranking and dense vector components.

## Overview 
For Assignment 2, we extended our Assignment 1 IR system by incorporating neural retrieval methods. Two main approaches were implemented:

* Dense Retrieval:
Using a pre-trained Sentence-BERT model (default: all-MiniLM-L6-v2), we encode both documents and queries into dense vector representations. Documents are then ranked by computing the cosine similarity between the query and document embeddings.

* BERT Re-Ranking:
We first use our baseline TF-IDF retrieval system (from Assignment 1) to generate a candidate list of up to 100 documents for each query. Then, we re-rank these candidates with a pre-trained CrossEncoder model (default: cross-encoder/ms-marco-MiniLM-L-6-v2) that computes a refined similarity score for each query–document pair.

Both methods were implemented in a new Python file (assignment2.py) so that the original Assignment 1 code remains unchanged.


## Program Functionality
* Preprocessing:
The system uses NLTK to tokenize text, remove punctuation and digits, filter out stopwords, discard words with fewer than three characters, and apply the Porter stemmer. In addition to the preprocessed tokens, the original text is retained (stored as “raw”) to support neural retrieval methods.

* Data Loading:
The corpus (in JSONL format) is loaded with each document’s _id, title, and text fields concatenated. Queries (also in JSONL format) are read and only those with relevance judgments are retained. Relevance annotations (from a TSV file) indicate which documents are relevant for each query.

* Indexing and Baseline Retrieval:
An inverted index is built from the preprocessed tokens. For each query, a TF-IDF vector is computed and candidate documents (those containing at least one query term) are scored using cosine similarity. The top 100 results are output in TREC format.

* Dense Retrieval:
The system uses the SentenceTransformer to generate dense embeddings for both the documents (using the original text) and the queries. Cosine similarity between these embeddings is computed to rank documents directly.

* BERT Re-Ranking:
The baseline TF-IDF method is first used to produce a candidate set of documents. Then, each query–document pair is passed to a CrossEncoder model to compute a refined relevance score, and the candidates are re-ranked accordingly.

* Output:
The results for each query are written in TREC format (i.e., query_id Q0 doc_id rank score run_tag) to a file. Our best system’s results (from Dense Retrieval) are submitted.


## How to Run
Dependencies:
Ensure Python 3 is installed. The following Python packages are required:
* nltk
* sentence-transformers
* numpy
  
**You can install these using pip**:
* pip install nltk sentence-transformers numpy
  
**Running the System:**
  The system is invoked from the command line：

  * ![image](https://github.com/user-attachments/assets/31d69674-0485-4e51-a924-c86c582d3815)


（To run the BERT Re-Ranking method, replace --method dense with --method bert.）
  *  Sample run of dense re-ranking, around 3 minutes for run:
     ![image](https://github.com/user-attachments/assets/7d0f3212-d505-44bf-916e-6093e44f6c1b)

  *  Sample run of bert re-ranking, around 25 minutes for run:
     ![image](https://github.com/user-attachments/assets/f08b3659-2a62-4f41-b624-2e22bc573a3a)


**Evaluation Process Using trec_eval**:
```
trec_eval formated-test.tsv Results_bert.txt

```

and 

```
trec_eval formated-test.tsv Results_dense.txt

```
 **Alternatively,you can also run python evaluate.py in the terminal**
- Using the Python version of TREC eval (pytrec_eval) is strongly recommended because it easily integrates into your Python pipeline, avoids compilation or system compatibility issues, and provides fast, accurate evaluation of IR metrics directly from your scripts.
 ```
 python evaluate.py
```
 ![image](https://github.com/user-attachments/assets/6efc6284-1df0-4496-a3af-c68b7f1fe7b9)


**Output:**

The program outputs a results file (e.g., Results_dense.txt for Dense Retrieval， Results_Bert.txt for Bert Retrievak) in TREC format. This file, along with our report, is included in the submission.

## Algorithms, Data Structures, and Optimizations

- Preprocessing:
   * Algorithms: Standard tokenization, stopword removal, and Porter stemming.
   * Data Structures: Python lists for tokens and sets for stopwords; dictionaries to store document data.
   * Optimizations: Converting text to lowercase and filtering out short tokens to reduce noise.
     
-Indexing and Baseline Retrieval:
   * Algorithms: Building an inverted index to quickly identify candidate documents. TF and IDF computations are used to produce TF-IDF vectors.
   * Data Structures: Dictionaries (or defaultdict) to map terms to sets of document IDs; dictionaries to store vector representations.
   * Optimizations: Limiting cosine similarity computations only to candidate documents (those sharing at least one query term).

-Dense Retrieval:
   * Algorithms: Dense vector representations are generated using a pre-trained Sentence-BERT model. Cosine similarity is computed between query and document embeddings.
   * Data Structures: Numpy arrays for efficient numerical operations.
   * Optimizations: Use of pre-trained models that are optimized for speed (e.g., all-MiniLM-L6-v2), and batch encoding of document texts to reduce computational overhed.

- BERT Re-Ranking:
    * Algorithms: Candidate generation via baseline retrieval followed by scoring with a CrossEncoder that directly computes a relevance score for each query–document pair.
    * Data Structures: Python lists to construct query–document pairs; dictionaries to store re-ranked results.
    * Optimizations: Limiting the re-ranking process to a small candidate set (top 100) to reduce the number of expensive model inferences.
 

## Sample Output – First 10 Answers for Queries 1 and 3
![image](https://github.com/user-attachments/assets/3ab6da7c-326c-4b31-a634-2c9e909e5b52)
![image](https://github.com/user-attachments/assets/4334ba98-b8ed-41f6-8ac2-ec2e4787650e)


## Results and Discussion ：
- for the Dense Retrieval：

![image](https://github.com/user-attachments/assets/fb60bfae-ceb1-48b7-b279-8c5023f32f1a)

- for the Bert Retrieval:

![image](https://github.com/user-attachments/assets/3aa5b0be-d5a1-431f-8964-eee6cb7cb071)


-the results from assignment 1:
![image](https://github.com/user-attachments/assets/41b3833a-f313-4d04-8cd2-857b905477f8)

![image](https://github.com/user-attachments/assets/5231281c-fa74-4a28-87c5-c2e5b7e8d5db)


### Comparison with Assignment 1

In Assignment 1, our system relied on a traditional vector space model (e.g., TF–IDF) and achieved a MAP score of around 0.50 (see the second image in the screenshots). While this performance was satisfactory as a baseline, we aimed to improve ranking quality in Assignment 2 by integrating neural retrieval methods.

* BERT Retrieval (Assignment 2): Achieved a MAP of 0.6501, NDCG of 0.7128, Reciprocal Rank of 0.6648, and P@10 of 0.0923.

* Dense Re-Ranking (Assignment 2): Achieved a MAP of 0.6032, NDCG of 0.6768, Reciprocal Rank of 0.6111, and P@10 of 0.0883.

Both neural methods outperform the Assignment 1 system by a noticeable margin (up to +0.15 in MAP for Dense Retrieval). This improvement indicates that leveraging pre-trained language models and dense embeddings provides a stronger representation of semantic relationships between queries and documents than a purely term-based TF–IDF approach.


In particular, our evaluation shows that the BERT Re-Ranking approach produced the best overall MAP, indicating that re-scoring the candidate documents using a pre-trained CrossEncoder model can capture more refined semantic relationships between queries and documents. This improved MAP suggests that, when starting from a traditional candidate set, the neural re-ranking step with BERT more effectively identifies the most relevant documents compared to directly using dense embeddings. In contrast, although Dense Retrieval also yielded competitive results, its MAP value was slightly lower, highlighting the added benefit of a re-scoring process in enhancing retrieval precision.

Overall, these results confirm that incorporating neural methods leads to more accurate retrieval performance compared to the original vector space model system from Assignment 1.

## References: 
* TREC Eval: https://github.com/cvangysel/pytrec_eval
* Nogueira, R., & Cho, K. (2020). Passage Re-ranking with BERT. arXiv preprint arXiv:1901.04085.
* Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of EMNLP-IJCNLP.





