# Semantic Search with Sentence Transformers

## Project Overview
This project demonstrates how to perform semantic search using Sentence Transformers and cosine similarity. It allows you to find the most relevant sentences or passages in a given text corpus based on the semantic meaning of a query, rather than just keyword matching.

## Technologies Used
-   **Python**: The core programming language.
-   **`sentence-transformers`**: A Python library for state-of-the-art sentence, paragraph, and image embeddings.
    -   **Model**: `intfloat/e5-small-v2` is used for generating high-quality embeddings.
-   **`scikit-learn`**: A machine learning library used here for `cosine_similarity` to compare embeddings.
-   **`numpy`**: For numerical operations, especially with array manipulations.

## How it Works
The process involves converting text into numerical vectors (embeddings) and then using these vectors to find semantic similarities.

1.  **Embedding Generation**: Each sentence or passage is transformed into a high-dimensional vector using a pre-trained `SentenceTransformer` model. These embeddings capture the semantic meaning of the text.
2.  **Query Embedding**: The user's query is also converted into an embedding using the same model.
3.  **Cosine Similarity**: The similarity between the query embedding and each passage embedding is calculated using cosine similarity. A higher cosine similarity score indicates greater semantic resemblance.
4.  **Ranking and Retrieval**: Passages are ranked based on their similarity scores to the query, and the top-ranked passages are retrieved as the most relevant results.

## Setup and Installation
To run this project, you need to install the required libraries. You can do this using pip:

```bash
pip install sentence-transformers scikit-learn numpy
```

## Usage Example

The notebook `Vectorization-with-Hugging-Face.ipynb` contains the full implementation. Here's a conceptual breakdown:

1.  **Load the Model**:
    ```python
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/e5-small-v2")
    ```

2.  **Define Passages**:
    A list of sentences or text passages that form your knowledge base.
    ```python
sentences = [
    "Artificial Intelligence is transforming healthcare by improving diagnostics.",
    "Machine learning models can detect patterns in large datasets.",
    "Natural Language Processing helps machines understand human language.",
    "Deep learning is a subset of machine learning using neural networks.",
    "AI systems require large amounts of quality data for training."
]
    ```

3.  **Generate Passage Embeddings**:
    Each passage is converted into its numerical representation. Note the `passage:` prefix for e5 models.
    ```python
    passages = [f"passage: {s}" for s in sentences]
    passage_embeddings = model.encode(passages, normalize_embeddings=True)
    ```

4.  **Define a Query and Get its Embedding**:
    The query is also embedded. Note the `query:` prefix for e5 models.
    ```python
    query = "How do machines understand human language?"
    query_embedding = model.encode(f"query: {query}", normalize_embeddings=True)
    ```

5.  **Calculate Similarity and Find Top Matches**:
    ```python
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity([query_embedding], passage_embeddings)[0]
    
    # Get best match
    top_index = np.argmax(scores)
    print("Most Relevant Sentence:")
    print(sentences[top_index])
    print("Similarity Score:", scores[top_index])

    # Get top K matches
    top_k = 2
    top_indices = scores.argsort()[-top_k:][::-1]
    print("\nTop Relevant Sentences:")
    for idx in top_indices:
        print(f"- {sentences[idx]}  (score: {scores[idx]:.4f})")
    ```

## Example Output
For the query "How do machines understand human language?", the output might look like this:

```
Query: How do machines understand human language?

Most Relevant Sentence:
Natural Language Processing helps machines understand human language.

Similarity Score: 0.9037678

Top Relevant Sentences:
- Natural Language Processing helps machines understand human language.  (score: 0.9038)
- Deep learning is a subset of machine learning using neural networks.  (score: 0.8497)
```

This output clearly shows how the semantic search identifies the most contextually relevant sentences.
