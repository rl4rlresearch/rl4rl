MECHANISM: Tied-embedding token/position translation gauge quotient

HYPOTHESIS: Fixing a sixth token/position translation coordinate will reduce the verified 1,597-parameter model to 1,596 parameters while retaining at least 99% accuracy, because the input embeddings remain unchanged and the tied output embedding changes logits only by a vocabulary-uniform offset.

INTENDED_EDIT: Extend `CompactPositionEmbedding` and its initialization from five to six fixed translation coordinates while preserving the centered positional-origin tail and positional rows 1–3.

EVIDENCE: The current design achieves 99.82% accuracy with five coordinatewise translation gauges at 1,597 parameters, whereas extending positional-row centering to row 4 fell to 97.25%; testing another instance of the already-used exact translation gauge is therefore the most informative next reduction.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Position embedding with five translations and four row-shift gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.fixed_coordinates = 5
=======
class CompactPositionEmbedding(nn.Module):
    """Position embedding with six translations and four row-shift gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.fixed_coordinates = 6
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix five token/position translations, then center the remaining
        # positional-origin tail and positional rows 1 and 2.
        with torch.no_grad():
            for coordinate in range(5):
                offset = self.pos_emb.weight[0, coordinate].detach().clone()
                self.token_emb.weight[:, coordinate].add_(offset)
                self.pos_emb.weight[:, coordinate].sub_(offset)

            origin_shift = -self.pos_emb.weight[0, 5:].mean()
            self.token_emb.weight[:, :5].add_(origin_shift)
            self.pos_emb.weight[:, :5].sub_(origin_shift)
            self.pos_emb.weight[0].add_(origin_shift)
=======
        # Fix six token/position translations, then center the remaining
        # positional-origin tail and positional rows 1 through 3.
        with torch.no_grad():
            for coordinate in range(6):
                offset = self.pos_emb.weight[0, coordinate].detach().clone()
                self.token_emb.weight[:, coordinate].add_(offset)
                self.pos_emb.weight[:, coordinate].sub_(offset)

            origin_shift = -self.pos_emb.weight[0, 6:].mean()
            self.token_emb.weight[:, :6].add_(origin_shift)
            self.pos_emb.weight[:, :6].sub_(origin_shift)
            self.pos_emb.weight[0].add_(origin_shift)
>>>>>>> REPLACE