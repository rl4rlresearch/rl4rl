MECHANISM: Four-dimensional tied token bottleneck

HYPOTHESIS: Reducing the tied token representation from five to four learned channels will lower the model from 1085 to 971 parameters while retaining at least 99% accuracy, because the five-channel design achieved 99.96% accuracy and the full eight-dimensional attention and MLP computation remains unchanged.

INTENDED_EDIT: Pad four-dimensional learned token embeddings with four fixed zero channels before transformer processing and use the same materialized vectors for tied output logits.

EVIDENCE: The immediately preceding reduction from six learned token channels to five achieved 99.96% accuracy at 1085 parameters, so the adjacent four-channel bottleneck is the most directly supported test of further token-representation redundancy.

<<<<<<< SEARCH
        if embedding_dim <= 3:
            raise ValueError("embedding_dim must exceed the three-channel bottleneck")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compact_dim = embedding_dim - 3
=======
        if embedding_dim <= 4:
            raise ValueError("embedding_dim must exceed the four-channel bottleneck")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compact_dim = embedding_dim - 4
>>>>>>> REPLACE