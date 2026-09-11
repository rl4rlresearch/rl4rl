MECHANISM: Joint token-position common-offset gauge fixing

HYPOTHESIS: Fixing the first shared embedding anchor at zero will reduce the model from 1605 to 1604 parameters while retaining at least 99% accuracy, because subtracting the same scalar from every token and position embedding only adds a common-coordinate shift before LayerNorm and a vocabulary-wide common logit offset.

INTENDED_EDIT: Replace the eight learned embedding anchors with seven learned anchors and one fixed-zero anchor, applying the corresponding function-preserving common shift during initialization.

EVIDENCE: The current 1605-parameter design achieved 99.95% accuracy while already quotienting eight token-position embedding gauges; the remaining simultaneous common offset is a distinct exact redundancy that does not further constrain the sensitive LayerNorm or MLP-bias coordinates.

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
    """Token and position tables with seven learned anchors and one fixed anchor."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.anchor_rest = nn.Parameter(torch.empty(7))
        self.token_rest = nn.Parameter(torch.empty(vocab_size * d_model - 8))
        self.pos_rest = nn.Parameter(torch.empty(max_seq_len * d_model - 8))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def token_weight(self) -> torch.Tensor:
        fourth_index = self.d_model + 3
=======
    def token_weight(self) -> torch.Tensor:
        anchor = torch.cat((self.anchor_rest.new_zeros(1), self.anchor_rest))
        fourth_index = self.d_model + 3
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
                anchor[:3],
                self.token_rest[:fourth_offset],
                anchor[3:4],
                self.token_rest[fourth_offset:fifth_offset],
                anchor[4:5],
                self.token_rest[fifth_offset:sixth_offset],
                anchor[5:6],
                self.token_rest[sixth_offset:seventh_offset],
                anchor[6:7],
                self.token_rest[seventh_offset:eighth_offset],
                anchor[7:],
>>>>>>> REPLACE

<<<<<<< SEARCH
    def pos_weight(self) -> torch.Tensor:
        # The third through eighth anchors train only through their token uses,
        # avoiding additional positional-gradient coupling.
        pos_anchor = torch.cat((self.anchor[:2], self.anchor[2:].detach()))
        return torch.cat((pos_anchor, self.pos_rest)).view(self.max_seq_len, self.d_model)
=======
    def pos_weight(self) -> torch.Tensor:
        # The third through eighth anchors train only through their token uses,
        # avoiding additional positional-gradient coupling.
        anchor = torch.cat((self.anchor_rest.new_zeros(1), self.anchor_rest))
        pos_anchor = torch.cat((anchor[:2], anchor[2:].detach()))
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

            # A common shift of both tables is invisible to every LayerNorm,
            # while tied-output logits change only by a class-independent offset.
            common_shift = token[0, 0].clone()
            token.sub_(common_shift)
            pos.sub_(common_shift)

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
                module.anchor_rest.copy_(
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