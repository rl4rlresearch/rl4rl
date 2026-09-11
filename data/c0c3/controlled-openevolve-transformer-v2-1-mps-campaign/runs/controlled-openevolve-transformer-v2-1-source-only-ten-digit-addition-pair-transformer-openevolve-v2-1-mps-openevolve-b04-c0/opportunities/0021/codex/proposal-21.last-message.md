MECHANISM: Distributed fifth one-sided embedding gauge anchor

HYPOTHESIS: Adding a detached-position fifth anchor between `token[2, 4]` and `position[0, 4]` will reduce the model from 1624 to 1623 parameters while retaining at least 99% accuracy, because distributing the fourth anchor to a separate token row restored accuracy to 99.98%, and this applies the same exact gauge symmetry without concentrating another constraint on an existing endpoint.

INTENDED_EDIT: Expand the shared embedding representation to five anchors, placing the new coordinate on token row 2 and detaching its positional gradient while preserving gauge-equivalent initialization.

EVIDENCE: Four anchors concentrated on token row zero achieved only 74.22%, whereas moving the fourth endpoint to token row one achieved 99.98% at 1624 parameters; distributing the next anchor to token row two is the smallest evidence-supported reduction.

<<<<<<< SEARCH
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
=======
class SharedAnchorEmbeddings(nn.Module):
    """Token and position tables sharing five gauge-redundant scalars."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.anchor = nn.Parameter(torch.empty(5))
        self.token_rest = nn.Parameter(torch.empty(vocab_size * d_model - 5))
        self.pos_rest = nn.Parameter(torch.empty(max_seq_len * d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    def token_weight(self) -> torch.Tensor:
        fourth_index = self.d_model + 3
        fifth_index = 2 * self.d_model + 4
        fourth_offset = fourth_index - 3
        fifth_offset = fourth_offset + fifth_index - fourth_index - 1
        flat = torch.cat(
            (
                self.anchor[:3],
                self.token_rest[:fourth_offset],
                self.anchor[3:4],
                self.token_rest[fourth_offset:fifth_offset],
                self.anchor[4:],
                self.token_rest[fifth_offset:],
            )
        )
        return flat.view(self.vocab_size, self.d_model)

    def pos_weight(self) -> torch.Tensor:
        # The third through fifth anchors train only through their token uses,
        # avoiding additional positional-gradient coupling.
        pos_anchor = torch.cat((self.anchor[:2], self.anchor[2:].detach()))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            fourth_shift = 0.5 * (pos[0, 3] - token[1, 3])
            token[:, 3].add_(fourth_shift)
            pos[:, 3].sub_(fourth_shift)

            fifth_shift = 0.5 * (pos[0, 4] - token[2, 4])
            token[:, 4].add_(fifth_shift)
            pos[:, 4].sub_(fifth_shift)

            fourth_index = module.d_model + 3
            fifth_index = 2 * module.d_model + 4
            token_flat = token.flatten()
            with torch.no_grad():
                module.anchor.copy_(
                    torch.cat((token[0, :3], token[1, 3:4], token[2, 4:5]))
                )
                module.token_rest.copy_(
                    torch.cat(
                        (
                            token_flat[3:fourth_index],
                            token_flat[fourth_index + 1 : fifth_index],
                            token_flat[fifth_index + 1 :],
                        )
                    )
                )
                module.pos_rest.copy_(pos.flatten()[5:])
>>>>>>> REPLACE