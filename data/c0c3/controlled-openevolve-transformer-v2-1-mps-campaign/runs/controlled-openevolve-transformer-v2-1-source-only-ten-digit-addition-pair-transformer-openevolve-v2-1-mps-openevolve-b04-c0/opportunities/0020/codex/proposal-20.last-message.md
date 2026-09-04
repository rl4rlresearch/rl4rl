MECHANISM: Distributed token-endpoint gauge sharing

HYPOTHESIS: Anchoring the fourth gauge coordinate to `token[1, 3]` instead of concentrating four constraints on token zero will reduce the model to 1624 parameters while retaining at least 99% accuracy, because it preserves the successful position-zero, token-only-gradient mechanism while distributing embedding optimization across token rows.

INTENDED_EDIT: Keep the three successful anchors unchanged and add a detached-position fourth anchor between `token[1, 3]` and `position[0, 3]`, with a gauge-equivalent initialization.

EVIDENCE: Three anchors with the third positional gradient detached achieved 99.95% at 1625 parameters, while placing a fourth such anchor on token zero fell to 74.22%; the severe failure after concentrating another constraint on the same token row motivates changing only the fourth token endpoint.

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

        # Match the random-number consumption of two nn.Embedding constructors.
        nn.init.normal_(torch.empty(vocab_size, d_model))
        nn.init.normal_(torch.empty(max_seq_len, d_model))

    def token_weight(self) -> torch.Tensor:
        fourth_index = self.d_model + 3
        before_fourth = self.token_rest[: fourth_index - 3]
        after_fourth = self.token_rest[fourth_index - 3 :]
        flat = torch.cat((self.anchor[:3], before_fourth, self.anchor[3:], after_fourth))
        return flat.view(self.vocab_size, self.d_model)

    def pos_weight(self) -> torch.Tensor:
        # The third and fourth anchors train only through their token uses,
        # avoiding additional positional-gradient coupling.
        pos_anchor = torch.cat((self.anchor[:2], self.anchor[2:].detach()))
        return torch.cat((pos_anchor, self.pos_rest)).view(self.max_seq_len, self.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
            # Gauge-transform the ordinary initialization so the two anchors
            # agree while token-plus-position inputs remain unchanged.
            shift = 0.5 * (pos[0, :3] - token[0, :3])
            token[:, :3].add_(shift)
            pos[:, :3].sub_(shift)
            with torch.no_grad():
                module.anchor.copy_(token[0, :3].reshape_as(module.anchor))
                module.token_rest.copy_(token.flatten()[3:])
                module.pos_rest.copy_(pos.flatten()[3:])
=======
            # Gauge-transform the ordinary initialization so each anchor pair
            # agrees while token-plus-position inputs remain unchanged.
            shift = 0.5 * (pos[0, :3] - token[0, :3])
            token[:, :3].add_(shift)
            pos[:, :3].sub_(shift)

            fourth_shift = 0.5 * (pos[0, 3] - token[1, 3])
            token[:, 3].add_(fourth_shift)
            pos[:, 3].sub_(fourth_shift)

            fourth_index = module.d_model + 3
            token_flat = token.flatten()
            with torch.no_grad():
                module.anchor.copy_(torch.cat((token[0, :3], token[1, 3:4])))
                module.token_rest.copy_(
                    torch.cat((token_flat[3:fourth_index], token_flat[fourth_index + 1 :]))
                )
                module.pos_rest.copy_(pos.flatten()[4:])
>>>>>>> REPLACE