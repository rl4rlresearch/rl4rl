MECHANISM: Global embedding-shift gauge fixing

HYPOTHESIS: Correctly removing the exact scalar shift symmetry shared by both embedding tables will reduce the model from 1562 to 1561 parameters while retaining at least 99% accuracy, because all normalized hidden states and output softmax probabilities remain invariant.

INTENDED_EDIT: Fix the first shared embedding anchor at zero, reindex the seven remaining learned anchors, and gauge-transform initialization by subtracting the former anchor from every token and position embedding entry.

EVIDENCE: The 1562-parameter design achieved 0.9999 accuracy, while extra attention and projection anchors fell to 0.721 and 0.9313. The prior embedding-shift attempt did not pass submission checks, so this exact redundancy remains untested and is the most informative next reduction.

<<<<<<< SEARCH
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
=======
class SharedAnchorEmbeddings(nn.Module):
    """Embedding tables with seven shared scalars and one fixed shift anchor."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.anchor = nn.Parameter(torch.empty(7))
        self.token_rest = nn.Parameter(torch.empty(vocab_size * d_model - 8))
        self.pos_rest = nn.Parameter(torch.empty(max_seq_len * d_model - 8))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        eighth_offset = seventh_offset + eighth_index - seventh_index - 1
        zero = self.token_rest.new_zeros(1)
        flat = torch.cat(
            (
                zero,
                self.anchor[:2],
                self.token_rest[:fourth_offset],
                self.anchor[2:3],
                self.token_rest[fourth_offset:fifth_offset],
                self.anchor[3:4],
                self.token_rest[fifth_offset:sixth_offset],
                self.anchor[4:5],
                self.token_rest[sixth_offset:seventh_offset],
                self.anchor[5:6],
                self.token_rest[seventh_offset:eighth_offset],
                self.anchor[6:],
                self.token_rest[eighth_offset:],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def pos_weight(self) -> torch.Tensor:
        # The third through eighth anchors train only through their token uses,
        # avoiding additional positional-gradient coupling.
        pos_anchor = torch.cat((self.anchor[:2], self.anchor[2:].detach()))
        return torch.cat((pos_anchor, self.pos_rest)).view(self.max_seq_len, self.d_model)
=======
    def pos_weight(self) -> torch.Tensor:
        # The final six learned anchors train only through their token uses,
        # avoiding additional positional-gradient coupling.
        zero = self.anchor.new_zeros(1)
        pos_anchor = torch.cat((zero, self.anchor[:1], self.anchor[1:].detach()))
        return torch.cat((pos_anchor, self.pos_rest)).view(self.max_seq_len, self.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
            eighth_shift = 0.5 * (pos[0, 7] - token[5, 7])
            token[:, 7].add_(eighth_shift)
            pos[:, 7].sub_(eighth_shift)

            fourth_index = module.d_model + 3
=======
            eighth_shift = 0.5 * (pos[0, 7] - token[5, 7])
            token[:, 7].add_(eighth_shift)
            pos[:, 7].sub_(eighth_shift)

            # A common scalar shift of both tables only shifts every residual
            # feature uniformly, which LayerNorm removes. The tied output
            # table then changes all logits by the same softmax-invariant value.
            global_shift = token[0, 0].clone()
            token.sub_(global_shift)
            pos.sub_(global_shift)

            fourth_index = module.d_model + 3
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.anchor.copy_(
                    torch.cat(
                        (
                            token[0, :3],
                            token[1, 3:4],
                            token[2, 4:5],
                            token[3, 5:6],
                            token[4, 6:7],
                            token[5, 7:8],
                        )
                    )
                )
=======
                module.anchor.copy_(
                    torch.cat(
                        (
                            token[0, 1:3],
                            token[1, 3:4],
                            token[2, 4:5],
                            token[3, 5:6],
                            token[4, 6:7],
                            token[5, 7:8],
                        )
                    )
                )
>>>>>>> REPLACE