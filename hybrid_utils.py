import numpy as np


def retrieval_probabilities(
    train_embeddings,
    query_embeddings,
    train_labels,
    num_classes,
    top_k=3,
    temperature=0.07,
    batch_size=128,
):
    """
    Convert top-k semantic neighbours into queue probability distributions.
    Inputs should contain normalized embeddings so dot product equals cosine similarity.
    """
    train_embeddings = np.asarray(train_embeddings, dtype=np.float32)
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int64)

    if len(train_embeddings) != len(train_labels):
        raise ValueError("Training embeddings and labels have different row counts.")
    if len(train_embeddings) == 0:
        raise ValueError("No historical ticket embeddings were supplied.")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")

    k = min(int(top_k), len(train_embeddings))
    all_probabilities = np.zeros(
        (len(query_embeddings), num_classes),
        dtype=np.float32,
    )
    all_indices = np.zeros((len(query_embeddings), k), dtype=np.int64)

    for start in range(0, len(query_embeddings), batch_size):
        end = min(start + batch_size, len(query_embeddings))
        similarities = query_embeddings[start:end] @ train_embeddings.T

        top_indices = np.argpartition(
            similarities,
            kth=similarities.shape[1] - k,
            axis=1,
        )[:, -k:]
        top_scores = np.take_along_axis(similarities, top_indices, axis=1)

        order = np.argsort(top_scores, axis=1)[:, ::-1]
        top_indices = np.take_along_axis(top_indices, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)

        scaled = top_scores / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        weights = np.exp(scaled)
        weights /= weights.sum(axis=1, keepdims=True)

        neighbor_labels = train_labels[top_indices]
        probabilities = np.zeros((end - start, num_classes), dtype=np.float32)

        for row in range(end - start):
            np.add.at(probabilities[row], neighbor_labels[row], weights[row])

        all_probabilities[start:end] = probabilities
        all_indices[start:end] = top_indices

    return all_probabilities, all_indices
