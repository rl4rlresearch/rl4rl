MECHANISM: Three-dimensional tied token bottleneck

HYPOTHESIS: Reducing the tied token representation from four to three learned channels will lower the 971-parameter model by one parameter per vocabulary token while retaining at least 99% accuracy, because the four-channel bottleneck achieved 99.83% accuracy and the full eight-dimensional attention and MLP computation remains unchanged.

INTENDED_EDIT: Pad three-dimensional learned token embeddings with five fixed zero channels before transformer processing and use the same materialized vectors for tied output logits.

EVIDENCE: The immediately preceding reduction from five learned token channels to four achieved 99.83% accuracy at 971 parameters, so the adjacent three-channel bottleneck is the most direct test of remaining token-representation redundancy.

<<<<<<< SEARCH
        if embedding_dim <= 4:
            raise ValueError("embedding_dim must exceed the four-channel bottleneck")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compact_dim = embedding_dim - 4
=======
        if embedding_dim <= 5:
            raise ValueError("embedding_dim must exceed the five-channel bottleneck")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compact_dim = embedding_dim - 5
>>>>>>> REPLACE