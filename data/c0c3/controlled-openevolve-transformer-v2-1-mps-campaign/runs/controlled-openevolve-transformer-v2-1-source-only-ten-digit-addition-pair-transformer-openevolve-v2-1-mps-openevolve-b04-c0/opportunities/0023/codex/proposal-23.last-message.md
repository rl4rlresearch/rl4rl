MECHANISM: Distributed seventh one-sided embedding gauge anchor

HYPOTHESIS: Adding a detached-position seventh anchor between `token[4, 6]` and `position[0, 6]` will reduce the model from 1622 to 1621 parameters while retaining at least 99% accuracy, because the similarly distributed fourth, fifth, and sixth anchors each preserved accuracy above 99.9%.

INTENDED_EDIT: Expand the shared embedding representation to seven anchors, place the new coordinate on token row 4, detach its positional gradient, and preserve gauge-equivalent initialization.

EVIDENCE: The sixth detached anchor on a previously unused token row achieved 99.97% at 1622 parameters, continuing the success of the distributed fourth and fifth anchors; this motivates applying the same mechanism to the next unused token row and coordinate.

<<<<<<< SEARCH
class SharedAnchorEmbeddings(nn.Module):
    """Token and position tables sharing six gauge-redundant scalars."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.anchor = nn.Parameter(torch.empty(6))
        self.token_rest = nn.Parameter(torch.empty(vocab_size * d_model - 6))
        self.pos_rest = nn.Parameter(torch.empty(max_seq_len * d_model - 6))
=======
class SharedAnchorEmbeddings(nn.Module):
    """Token and position tables sharing seven gauge-redundant scalars."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.anchor = nn.Parameter(torch.empty(7))
        self.token_rest = nn.Parameter(torch.empty(vocab_size * d_model - 7))
        self.pos_rest = nn.Parameter(torch.empty(max_seq_len * d_model - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def token_weight(self) -> torch.Tensor:
        fourth_index = self.d_model + 3
        fifth_index = 2 * self.d_model + 4
        sixth_index = 3 * self.d_model + 5
        fourth_offset = fourth_index - 3
        fifth_offset = fourth_offset + fifth_index - fourth_index - 1
        sixth_offset = fifth_offset + sixth_index - fifth_index - 1
        flat = torch.cat(
            (
                self.anchor[:3],
                self.token_rest[:fourth_offset],
                self.anchor[3:4],
                self.token_rest[fourth_offset:fifth_offset],
                self.anchor[4:5],
                self.token_rest[fifth_offset:sixth_offset],
                self.anchor[5:],
                self.token_rest[sixth_offset:],
            )
        )
        return flat.view(self.vocab_size, self.d_model)

    def pos_weight(self) -> torch.Tensor:
        # The third through sixth anchors train only through their token uses,
        # avoiding additional positional-gradient coupling.
        pos_anchor = torch.cat((self.anchor[:2], self.anchor[2:].detach()))
        return torch.cat((pos_anchor, self.pos_rest)).view(self.max_seq_len, self.d_model)
=======
    def token_weight(self) -> torch.Tensor:
        fourth_index = self.d_model + 3
        fifth_index = 2 * self.d_model + 4
        sixth_index = 3 * self.d_model + 5
        seventh_index = 4 * self.d_model + 6
        fourth_offset = fourth_index - 3
        fifth_offset = fourth_offset + fifth_index - fourth_index - 1
        sixth_offset = fifth_offset + sixth_index - fifth_index - 1
        seventh_offset = sixth_offset + seventh_index - sixth_index - 1
        flat = torch.cat(
            (
                self.anchor[:3],
                self.token_rest[:fourth_offset],
                self.anchor[3:4],
                self.token_rest[fourth_offset:fifth_offset],
                self.anchor[4:5],
                self.token_rest[fifth_offset:sixth_offset],
                self.anchor[5:6],
                self.token_rest[sixth_offset:seventh_offset],
                self.anchor[6:],
                self.token_rest[seventh_offset:],
            )
        )
        return flat.view(self.vocab_size, self.d_model)

    def pos_weight(self) -> torch.Tensor:
        # The third through seventh anchors train only through their token uses,
        # avoiding additional positional-gradient coupling.
        pos_anchor = torch.cat((self.anchor[:2], self.anchor[2:].detach()))
        return torch.cat((pos_anchor, self.pos_rest)).view(self.max_seq_len, self.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
            sixth_shift = 0.5 * (pos[0, 5] - token[3, 5])
            token[:, 5].add_(sixth_shift)
            pos[:, 5].sub_(sixth_shift)

            fourth_index = module.d_model + 3
            fifth_index = 2 * module.d_model + 4
            sixth_index = 3 * module.d_model + 5
            token_flat = token.flatten()
=======
            sixth_shift = 0.5 * (pos[0, 5] - token[3, 5])
            token[:, 5].add_(sixth_shift)
            pos[:, 5].sub_(sixth_shift)

            seventh_shift = 0.5 * (pos[0, 6] - token[4, 6])
            token[:, 6].add_(seventh_shift)
            pos[:, 6].sub_(seventh_shift)

            fourth_index = module.d_model + 3
            fifth_index = 2 * module.d_model + 4
            sixth_index = 3 * module.d_model + 5
            seventh_index = 4 * module.d_model + 6
            token_flat = token.flatten()
>>>>>>> REPLACE

<<<<<<< SEARCH
                            token[0, :3],
                            token[1, 3:4],
                            token[2, 4:5],
                            token[3, 5:6],
=======
                            token[0, :3],
                            token[1, 3:4],
                            token[2, 4:5],
                            token[3, 5:6],
                            token[4, 6:7],
>>>>>>> REPLACE

<<<<<<< SEARCH
                            token_flat[3:fourth_index],
                            token_flat[fourth_index + 1 : fifth_index],
                            token_flat[fifth_index + 1 : sixth_index],
                            token_flat[sixth_index + 1 :],
                        )
                    )
                )
                module.pos_rest.copy_(pos.flatten()[6:])
=======
                            token_flat[3:fourth_index],
                            token_flat[fourth_index + 1 : fifth_index],
                            token_flat[fifth_index + 1 : sixth_index],
                            token_flat[sixth_index + 1 : seventh_index],
                            token_flat[seventh_index + 1 :],
                        )
                    )
                )
                module.pos_rest.copy_(pos.flatten()[7:])
>>>>>>> REPLACE