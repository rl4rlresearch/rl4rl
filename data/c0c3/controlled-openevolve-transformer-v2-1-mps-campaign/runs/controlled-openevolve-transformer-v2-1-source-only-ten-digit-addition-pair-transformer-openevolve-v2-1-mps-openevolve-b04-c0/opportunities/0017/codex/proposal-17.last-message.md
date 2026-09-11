MECHANISM: Causally re-anchored embedding gauge sharing

HYPOTHESIS: Sharing a third token coordinate with the causally final positional embedding will reduce the model from 1626 to 1625 parameters while retaining at least 99% accuracy, because it preserves the exact token/position shift symmetry while coupling the new shared parameter to fewer downstream attention computations than position zero.

INTENDED_EDIT: Retain the two successful position-zero anchors and add a third anchor between `token[0, 2]` and the final position’s coordinate 2, with a gauge-equivalent initialization.

EVIDENCE: Two position-zero anchors achieved 100% at 1626 parameters, while a third reached 97.24%; relocating only the failing third constraint tests whether its optimization coupling to the causally most influential position caused the shortfall.

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
        self.third_pos_flat = (max_seq_len - 1) * d_model + 2
        self.anchor = nn.Parameter(torch.empty(3))
        self.token_rest = nn.Parameter(torch.empty(vocab_size * d_model - 3))
        self.pos_rest = nn.Parameter(torch.empty(max_seq_len * d_model - 3))

        # Match the random-number consumption of two nn.Embedding constructors.
        nn.init.normal_(torch.empty(vocab_size, d_model))
        nn.init.normal_(torch.empty(max_seq_len, d_model))

    def token_weight(self) -> torch.Tensor:
        return torch.cat((self.anchor, self.token_rest)).view(self.vocab_size, self.d_model)

    def pos_weight(self) -> torch.Tensor:
        split = self.third_pos_flat - 2
        flat = torch.cat(
            (
                self.anchor[:2],
                self.pos_rest[:split],
                self.anchor[2:3],
                self.pos_rest[split:],
            )
        )
        return flat.view(self.max_seq_len, self.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
            # Gauge-transform the ordinary initialization so the two anchors
            # agree while token-plus-position inputs remain unchanged.
            shift = 0.5 * (pos[0, :2] - token[0, :2])
            token[:, :2].add_(shift)
            pos[:, :2].sub_(shift)
            with torch.no_grad():
                module.anchor.copy_(token[0, :2].reshape_as(module.anchor))
                module.token_rest.copy_(token.flatten()[2:])
                module.pos_rest.copy_(pos.flatten()[2:])
=======
            # Gauge-transform the ordinary initialization so the first two
            # anchors use position zero and the third uses the final position.
            pos_anchor = torch.stack((pos[0, 0], pos[0, 1], pos[-1, 2]))
            shift = 0.5 * (pos_anchor - token[0, :3])
            token[:, :3].add_(shift)
            pos[:, :3].sub_(shift)
            pos_flat = pos.flatten()
            with torch.no_grad():
                module.anchor.copy_(token[0, :3].reshape_as(module.anchor))
                module.token_rest.copy_(token.flatten()[3:])
                module.pos_rest.copy_(
                    torch.cat(
                        (
                            pos_flat[2 : module.third_pos_flat],
                            pos_flat[module.third_pos_flat + 1 :],
                        )
                    )
                )
>>>>>>> REPLACE