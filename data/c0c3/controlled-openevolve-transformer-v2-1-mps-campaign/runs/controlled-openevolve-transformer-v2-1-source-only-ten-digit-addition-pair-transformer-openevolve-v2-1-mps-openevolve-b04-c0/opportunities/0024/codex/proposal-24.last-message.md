MECHANISM: Distributed eighth one-sided embedding gauge anchor

HYPOTHESIS: Adding a detached-position eighth anchor between `token[5, 7]` and `position[0, 7]` will reduce the model from 1621 to 1620 parameters while retaining at least 99% accuracy, because the similarly distributed fourth through seventh anchors all preserved accuracy above 99%.

INTENDED_EDIT: Expand the shared embedding representation to eight anchors, place the new coordinate on unused token row 5, detach its positional gradient, and preserve gauge-equivalent initialization.

EVIDENCE: The distributed seventh anchor achieved 99.72% accuracy at 1621 parameters, following 99.98%, 99.94%, and 99.97% results for the fourth through sixth distributed anchors; this supports applying the same mechanism to the final unused embedding coordinate.

<<<<<<< SEARCH
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
=======
class SharedAnchorEmbeddings(nn.Module):
    """Token and position tables sharing eight gauge-redundant scalars."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.anchor = nn.Parameter(torch.empty(8))
        self.token_rest = nn.Parameter(torch.empty(vocab_size * d_model - 8))
        self.pos_rest = nn.Parameter(torch.empty(max_seq_len * d_model - 8))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        seventh_index = 4 * self.d_model + 6
        eighth_index = 5 * self.d_model + 7
        fourth_offset = fourth_index - 3
        fifth_offset = fourth_offset + fifth_index - fourth_index - 1
        sixth_offset = fifth_offset + sixth_index - fifth_index - 1
        seventh_offset = sixth_offset + seventh_index - sixth_index - 1
        eighth_offset = seventh_offset + eighth_index - seventh_index - 1
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
                self.anchor[6:7],
                self.token_rest[seventh_offset:eighth_offset],
                self.anchor[7:],
                self.token_rest[eighth_offset:],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # The third through seventh anchors train only through their token uses,
=======
        # The third through eighth anchors train only through their token uses,
>>>>>>> REPLACE

<<<<<<< SEARCH
            seventh_shift = 0.5 * (pos[0, 6] - token[4, 6])
            token[:, 6].add_(seventh_shift)
            pos[:, 6].sub_(seventh_shift)

            fourth_index = module.d_model + 3
            fifth_index = 2 * module.d_model + 4
            sixth_index = 3 * module.d_model + 5
            seventh_index = 4 * module.d_model + 6
=======
            seventh_shift = 0.5 * (pos[0, 6] - token[4, 6])
            token[:, 6].add_(seventh_shift)
            pos[:, 6].sub_(seventh_shift)

            eighth_shift = 0.5 * (pos[0, 7] - token[5, 7])
            token[:, 7].add_(eighth_shift)
            pos[:, 7].sub_(eighth_shift)

            fourth_index = module.d_model + 3
            fifth_index = 2 * module.d_model + 4
            sixth_index = 3 * module.d_model + 5
            seventh_index = 4 * module.d_model + 6
            eighth_index = 5 * module.d_model + 7
>>>>>>> REPLACE

<<<<<<< SEARCH
                            token[3, 5:6],
                            token[4, 6:7],
                        )
                    )
                )
                module.token_rest.copy_(
                    torch.cat(
                        (
                            token_flat[3:fourth_index],
                            token_flat[fourth_index + 1 : fifth_index],
                            token_flat[fifth_index + 1 : sixth_index],
                            token_flat[sixth_index + 1 : seventh_index],
                            token_flat[seventh_index + 1 :],
                        )
                    )
                )
                module.pos_rest.copy_(pos.flatten()[7:])
=======
                            token[3, 5:6],
                            token[4, 6:7],
                            token[5, 7:8],
                        )
                    )
                )
                module.token_rest.copy_(
                    torch.cat(
                        (
                            token_flat[3:fourth_index],
                            token_flat[fourth_index + 1 : fifth_index],
                            token_flat[fifth_index + 1 : sixth_index],
                            token_flat[sixth_index + 1 : seventh_index],
                            token_flat[seventh_index + 1 : eighth_index],
                            token_flat[eighth_index + 1 :],
                        )
                    )
                )
                module.pos_rest.copy_(pos.flatten()[8:])
>>>>>>> REPLACE