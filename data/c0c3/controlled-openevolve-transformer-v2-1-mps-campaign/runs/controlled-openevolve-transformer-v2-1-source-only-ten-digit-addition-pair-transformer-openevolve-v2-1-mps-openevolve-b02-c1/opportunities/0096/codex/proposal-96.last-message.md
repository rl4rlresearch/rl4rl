MECHANISM: Extended terminal final-LayerNorm bias sharing

HYPOTHESIS: Sharing the final three nonreference LayerNorm bias coordinates will reduce the model from 989 to 988 parameters while retaining at least 99% accuracy, because terminal bias-pair sharing previously achieved 99.98% and this leaves the sensitive attention and MLP coordinates unchanged.

INTENDED_EDIT: Store five rather than six final-LayerNorm bias contrasts and reuse the fifth across the final three nonreference residual channels.

EVIDENCE: Terminal final-LayerNorm bias sharing achieved 99.98%, whereas recent failures involved additional key-tail removal or broader MLP sharing; extending the successful isolated bias constraint is the smallest orthogonal compression step.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(normalized_shape - 2))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 2))
=======
        self.weight = nn.Parameter(torch.ones(normalized_shape - 2))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat(
            [
                self.bias,
                self.bias[-1:],
                self.bias.new_zeros(1),
            ]
        )
=======
        bias = torch.cat(
            [
                self.bias,
                self.bias[-1:].expand(2),
                self.bias.new_zeros(1),
            ]
        )
>>>>>>> REPLACE