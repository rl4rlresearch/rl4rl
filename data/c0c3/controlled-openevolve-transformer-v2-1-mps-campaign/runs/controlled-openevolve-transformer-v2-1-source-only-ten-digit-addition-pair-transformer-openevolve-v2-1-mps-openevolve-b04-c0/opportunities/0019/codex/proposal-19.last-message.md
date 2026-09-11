MECHANISM: Fourth one-sided embedding gauge anchor

HYPOTHESIS: Extending the successful one-sided gauge sharing to a fourth embedding coordinate will reduce the model from 1625 to 1624 parameters while maintaining at least 99% accuracy, because the third detached positional pathway achieved 99.95% and the fourth applies the same exact symmetry without adding another coupled positional gradient.

INTENDED_EDIT: Share four token/position embedding coordinates, keeping the first two bidirectionally trained and detaching positional gradients for the third and fourth coordinates.

EVIDENCE: Three fully coupled anchors reached only 97.24%, whereas detaching the third anchor’s positional gradient reached 99.95% at 1625 parameters; this directly motivates extending the successful one-sided mechanism by one coordinate.

<<<<<<< SEARCH
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
=======
class SharedAnchorEmbeddings(nn.Module):
    """Token and position tables sharing four gauge-redundant scalars."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.anchor = nn.Parameter(torch.empty(4))
        self.token_rest = nn.Parameter(torch.empty(vocab_size * d_model - 4))
        self.pos_rest = nn.Parameter(torch.empty(max_seq_len * d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def pos_weight(self) -> torch.Tensor:
        # The third anchor follows the learned token value, but its positional
        # use does not add a second, potentially conflicting gradient pathway.
        pos_anchor = torch.cat((self.anchor[:2], self.anchor[2:].detach()))
        return torch.cat((pos_anchor, self.pos_rest)).view(self.max_seq_len, self.d_model)
=======
    def pos_weight(self) -> torch.Tensor:
        # Later anchors follow their learned token values, but their positional
        # uses do not add potentially conflicting gradient pathways.
        pos_anchor = torch.cat((self.anchor[:2], self.anchor[2:].detach()))
        return torch.cat((pos_anchor, self.pos_rest)).view(self.max_seq_len, self.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
            shift = 0.5 * (pos[0, :3] - token[0, :3])
            token[:, :3].add_(shift)
            pos[:, :3].sub_(shift)
            with torch.no_grad():
                module.anchor.copy_(token[0, :3].reshape_as(module.anchor))
                module.token_rest.copy_(token.flatten()[3:])
                module.pos_rest.copy_(pos.flatten()[3:])
=======
            shift = 0.5 * (pos[0, :4] - token[0, :4])
            token[:, :4].add_(shift)
            pos[:, :4].sub_(shift)
            with torch.no_grad():
                module.anchor.copy_(token[0, :4].reshape_as(module.anchor))
                module.token_rest.copy_(token.flatten()[4:])
                module.pos_rest.copy_(pos.flatten()[4:])
>>>>>>> REPLACE