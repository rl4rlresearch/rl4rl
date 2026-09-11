MECHANISM: One-sided third embedding gauge anchor

HYPOTHESIS: Expanding to three shared embedding coordinates while detaching only the third coordinate’s positional gradient will reduce the model from 1626 to 1625 parameters and achieve at least 99% accuracy, because it preserves the exact token/position gauge constraint but avoids the conflicting joint optimization pathway implicated by the coupled third anchor.

INTENDED_EDIT: Retain the two successful bidirectionally trained anchors, add a third shared coordinate initialized by the same gauge-preserving transformation, and train that new scalar only through the token/input-output embedding pathway.

EVIDENCE: Three fully coupled anchors reached 97.24% at 1625 parameters, close to the threshold, while moving the third coupling to the final position collapsed accuracy to 0.01%; this motivates preserving the exact third gauge reduction while isolating its positional-gradient coupling.

<<<<<<< SEARCH
class SharedAnchorEmbeddings(nn.Module):
    """Token and position tables sharing one gauge-redundant scalar."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.anchor = nn.Parameter(torch.empty(2))
        self.token_rest = nn.Parameter(torch.empty(vocab_size * d_model - 2))
        self.pos_rest = nn.Parameter(torch.empty(max_seq_len * d_model - 2))

        # Match the random-number consumption of two nn.Embedding constructors.
        nn.init.normal_(torch.empty(vocab_size, d_model))
        nn.init.normal_(torch.empty(max_seq_len, d_model))

    def token_weight(self) -> torch.Tensor:
        return torch.cat((self.anchor, self.token_rest)).view(self.vocab_size, self.d_model)

    def pos_weight(self) -> torch.Tensor:
        return torch.cat((self.anchor, self.pos_rest)).view(self.max_seq_len, self.d_model)
=======
class SharedAnchorEmbeddings(nn.Module):
    """Token and position tables sharing three gauge-redundant scalars."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.anchor = nn.Parameter(torch.empty(3))
        self.token_rest = nn.Parameter(torch.empty(vocab_size * d_model - 3))
        self.pos_rest = nn.Parameter(torch.empty(max_seq_len * d_model - 3))

        # Match the random-number consumption of two nn.Embedding constructors.
        nn.init.normal_(torch.empty(vocab_size, d_model))
        nn.init.normal_(torch.empty(max_seq_len, d_model))

    def token_weight(self) -> torch.Tensor:
        return torch.cat((self.anchor, self.token_rest)).view(self.vocab_size, self.d_model)

    def pos_weight(self) -> torch.Tensor:
        # The third anchor follows the learned token value, but its positional
        # use does not add a second, potentially conflicting gradient pathway.
        pos_anchor = torch.cat((self.anchor[:2], self.anchor[2:].detach()))
        return torch.cat((pos_anchor, self.pos_rest)).view(self.max_seq_len, self.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
            shift = 0.5 * (pos[0, :2] - token[0, :2])
            token[:, :2].add_(shift)
            pos[:, :2].sub_(shift)
            with torch.no_grad():
                module.anchor.copy_(token[0, :2].reshape_as(module.anchor))
                module.token_rest.copy_(token.flatten()[2:])
                module.pos_rest.copy_(pos.flatten()[2:])
=======
            shift = 0.5 * (pos[0, :3] - token[0, :3])
            token[:, :3].add_(shift)
            pos[:, :3].sub_(shift)
            with torch.no_grad():
                module.anchor.copy_(token[0, :3].reshape_as(module.anchor))
                module.token_rest.copy_(token.flatten()[3:])
                module.pos_rest.copy_(pos.flatten()[3:])
>>>>>>> REPLACE