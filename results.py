# Script to extract first 10 answers for Query 1 and Query 3 from Dense and BERT results

def print_first_10_answers(results_file, query_id):
    """
    Print first 10 lines from results_file that start with the given query_id.
    """
    with open(results_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Filter lines starting with the query id (followed by a space)
    query_lines = [line.strip() for line in lines if line.startswith(f"{query_id} ")]
    print(f"Results for Query {query_id} from {results_file}:")
    for line in query_lines[:10]:
        print(line)
    print("\n")


# Define file names
dense_results_file = "Results_dense.txt"
bert_results_file = "Results_bert.txt"

# For Dense Retrieval Results:
print_first_10_answers(dense_results_file, "1")
print_first_10_answers(dense_results_file, "3")

# For BERT Re-Ranking Results:
print_first_10_answers(bert_results_file, "1")
print_first_10_answers(bert_results_file, "3")
