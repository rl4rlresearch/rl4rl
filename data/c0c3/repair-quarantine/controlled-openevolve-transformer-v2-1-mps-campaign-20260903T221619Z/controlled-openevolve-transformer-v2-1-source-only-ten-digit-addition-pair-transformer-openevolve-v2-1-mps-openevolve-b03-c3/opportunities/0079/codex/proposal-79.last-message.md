MECHANISM: Eighth positional-row scalar-shift gauge with ambient AdamW

HYPOTHESIS: A 1,518-parameter model will retain at least 99% accuracy because the verified seven-gauge design achieved 99.98%, while the eighth positional row has the same exact pre-LayerNorm scalar-shift symmetry.

INTENDED_EDIT: Reproduce the qualified seven positional gauges, then gauge-fix the eighth row while preserving every omitted coordinate through initialization, gradient clipping, AdamW moments, decay, and updates.

EVIDENCE: Reference Design 1 achieved 99.98% accuracy with seven positional-row gauges at 1,519 parameters; every tested positional-gauge extension has remained above 99%, making the identical eighth-row reduction the strongest supported next test.

<<<<<<< SEARCH
class GaugeFixedPositionEmbedding(nn.Module):
    """Embedding with three independent positional scalar shifts removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.first = nn.Parameter(torch.empty(embedding_dim - 1))
        self.second = nn.Parameter(torch.empty(embedding_dim - 1))
        self.third = nn.Parameter(torch.empty(embedding_dim - 1))
        self.rest = nn.Parameter(
            torch.empty(num_embeddings - 3, embedding_dim)
        )
        self.full_first = None
        self.full_second = None
        self.full_third = None
        self.reset_parameters()
=======
class GaugeFixedPositionEmbedding(nn.Module):
    """Embedding with eight independent positional scalar shifts removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.first = nn.Parameter(torch.empty(embedding_dim - 1))
        self.second = nn.Parameter(torch.empty(embedding_dim - 1))
        self.third = nn.Parameter(torch.empty(embedding_dim - 1))
        self.fourth = nn.Parameter(torch.empty(embedding_dim - 1))
        self.fifth = nn.Parameter(torch.empty(embedding_dim - 1))
        self.sixth = nn.Parameter(torch.empty(embedding_dim - 1))
        self.seventh = nn.Parameter(torch.empty(embedding_dim - 1))
        self.eighth = nn.Parameter(torch.empty(embedding_dim - 1))
        self.rest = nn.Parameter(
            torch.empty(num_embeddings - 8, embedding_dim)
        )
        self.full_first = None
        self.full_second = None
        self.full_third = None
        self.full_fourth = None
        self.full_fifth = None
        self.full_sixth = None
        self.full_seventh = None
        self.full_eighth = None
        self.reset_parameters()
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.first.copy_(raw[0, :-1] - raw[0, -1])
        self.second.copy_(raw[1, :-1] - raw[1, -1])
        self.third.copy_(raw[2, :-1] - raw[2, -1])
        self.rest.copy_(raw[3:])
=======
        self.first.copy_(raw[0, :-1] - raw[0, -1])
        self.second.copy_(raw[1, :-1] - raw[1, -1])
        self.third.copy_(raw[2, :-1] - raw[2, -1])
        self.fourth.copy_(raw[3, :-1] - raw[3, -1])
        self.fifth.copy_(raw[4, :-1] - raw[4, -1])
        self.sixth.copy_(raw[5, :-1] - raw[5, -1])
        self.seventh.copy_(raw[6, :-1] - raw[6, -1])
        self.eighth.copy_(raw[7, :-1] - raw[7, -1])
        self.rest.copy_(raw[8:])
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first = torch.cat((self.first, self.first.new_zeros(1)))
        second = torch.cat((self.second, self.second.new_zeros(1)))
        third = torch.cat((self.third, self.third.new_zeros(1)))
        if torch.is_grad_enabled():
            first.retain_grad()
            second.retain_grad()
            third.retain_grad()
            self.full_first = first
            self.full_second = second
            self.full_third = third
        weight = torch.cat(
            (
                first.unsqueeze(0),
                second.unsqueeze(0),
                third.unsqueeze(0),
                self.rest,
            ),
            dim=0,
        )
        return F.embedding(idx, weight)
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first = torch.cat((self.first, self.first.new_zeros(1)))
        second = torch.cat((self.second, self.second.new_zeros(1)))
        third = torch.cat((self.third, self.third.new_zeros(1)))
        fourth = torch.cat((self.fourth, self.fourth.new_zeros(1)))
        fifth = torch.cat((self.fifth, self.fifth.new_zeros(1)))
        sixth = torch.cat((self.sixth, self.sixth.new_zeros(1)))
        seventh = torch.cat((self.seventh, self.seventh.new_zeros(1)))
        eighth = torch.cat((self.eighth, self.eighth.new_zeros(1)))
        if torch.is_grad_enabled():
            first.retain_grad()
            second.retain_grad()
            third.retain_grad()
            fourth.retain_grad()
            fifth.retain_grad()
            sixth.retain_grad()
            seventh.retain_grad()
            eighth.retain_grad()
            self.full_first = first
            self.full_second = second
            self.full_third = third
            self.full_fourth = fourth
            self.full_fifth = fifth
            self.full_sixth = sixth
            self.full_seventh = seventh
            self.full_eighth = eighth
        weight = torch.cat(
            (
                first.unsqueeze(0),
                second.unsqueeze(0),
                third.unsqueeze(0),
                fourth.unsqueeze(0),
                fifth.unsqueeze(0),
                sixth.unsqueeze(0),
                seventh.unsqueeze(0),
                eighth.unsqueeze(0),
                self.rest,
            ),
            dim=0,
        )
        return F.embedding(idx, weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_params = [
        model.token_emb.weight,
        model.pos_emb.first,
        model.pos_emb.second,
        model.pos_emb.third,
    ]
=======
    gauge_params = [
        model.token_emb.weight,
        model.pos_emb.first,
        model.pos_emb.second,
        model.pos_emb.third,
        model.pos_emb.fourth,
        model.pos_emb.fifth,
        model.pos_emb.sixth,
        model.pos_emb.seventh,
        model.pos_emb.eighth,
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
            model.pos_emb.full_first.grad.detach(),
            model.pos_emb.full_second.grad.detach(),
            model.pos_emb.full_third.grad.detach(),
        ]
=======
        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
            model.pos_emb.full_first.grad.detach(),
            model.pos_emb.full_second.grad.detach(),
            model.pos_emb.full_third.grad.detach(),
            model.pos_emb.full_fourth.grad.detach(),
            model.pos_emb.full_fifth.grad.detach(),
            model.pos_emb.full_sixth.grad.detach(),
            model.pos_emb.full_seventh.grad.detach(),
            model.pos_emb.full_eighth.grad.detach(),
        ]
>>>>>>> REPLACE