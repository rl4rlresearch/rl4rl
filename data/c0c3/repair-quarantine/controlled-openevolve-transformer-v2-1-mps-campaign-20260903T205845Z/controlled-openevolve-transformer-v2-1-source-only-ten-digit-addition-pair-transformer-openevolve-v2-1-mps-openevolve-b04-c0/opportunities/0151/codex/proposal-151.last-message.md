MECHANISM: Global embedding-shift gauge fixing

HYPOTHESIS: Fixing the first shared embedding anchor at zero will reduce the model from 1562 to 1561 learned parameters while retaining at least 99% accuracy, because a uniform shift of both embedding tables is removed by LayerNorm and changes tied-output logits only by a softmax-invariant common offset.

INTENDED_EDIT: Replace one learned shared embedding anchor with a fixed zero and gauge-transform initialization by subtracting that anchor from every token and position embedding entry.

EVIDENCE: The 1562-parameter design achieved 0.9999 accuracy, while extending attention anchoring reached only 0.721 and extending projection anchoring reached 0.9313; this motivates removing a separate exact embedding redundancy instead of further constraining those sensitive modules.

<<<<<<< SEARCH
class SharedAnchorEmbeddings(nn.Module):
    """Token and position tables sharing eight gauge-redundant scalars."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.anchor = nn.Parameter(torch.empty(8))
=======
class SharedAnchorEmbeddings(nn.Module):
    """Token and position tables with one fixed global-shift gauge."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.anchor = nn.Parameter(torch.empty(7))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        return flat.view(self.vocab_size, self.d_model)

    def pos_weight(self) -> torch.Tensor:
        # The third through eighth anchors train only through their token uses,
        # avoiding additional positional-gradient coupling.
        pos_anchor = torch.cat((self.anchor[:2], self.anchor[2:].detach()))
        return torch.cat((pos_anchor, self.pos_rest)).view(self.max_seq_len, self.d_model)
=======
        zero = self.anchor.new_zeros(1)
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
        return flat.view(self.vocab_size, self.d_model)

    def pos_weight(self) -> torch.Tensor:
        # The first reconstructed anchor is fixed. The third through eighth
        # train only through their token uses.
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

            # A common scalar shift of both tables produces only a
            # feature-uniform residual shift, while its tied-output effect is
            # a vocabulary-uniform logit offset. Fix that exact gauge at zero.
            global_offset = token[0, 0].clone()
            token.sub_(global_offset)
            pos.sub_(global_offset)
            token[0, 0] = 0.0
            pos[0, 0] = 0.0

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