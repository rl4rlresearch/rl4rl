MECHANISM: Gauge-fixed headwise Q/K factorization

HYPOTHESIS: Removing each head’s exact GL(4) query/key basis redundancy will reduce the model from 1,582 to 1,550 parameters while retaining at least 99% accuracy, because it preserves the full learned attention-score family and exactly preserves conventional Q/K initialization logits.

INTENDED_EDIT: Replace unconstrained query and key projections with a gauge-fixed factorization whose query anchor submatrix is fixed and whose learned key factor absorbs the inverse basis transformation; retain independent values, query biases, heads, and all other successful settings.

EVIDENCE: The 1,582-parameter model reaches 99.95% with full Q/K factors, while deleting one key direction fell to 41.04%; this instead removes only 32 functionally redundant factor-basis coordinates without deleting an addressing direction.

<<<<<<< SEARCH
import math
from dataclasses import dataclass
=======
import itertools
import math
from dataclasses import dataclass
>>>>>>> REPLACE

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class GaugeFixedQKV(nn.Module):
    """Q/K factors modulo each head's functionally redundant internal basis."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q_scale = 0.02

        # Preserve the constructor RNG consumption of the replaced QKV Linear.
        nn.Linear(d_model, 3 * d_model)

        free_dim = d_model - self.head_dim
        self.q_free = nn.Parameter(
            torch.empty(n_head, self.head_dim, free_dim)
        )
        self.k_weight = nn.Parameter(
            torch.empty(n_head, self.head_dim, d_model)
        )
        self.v_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.q_bias = nn.Parameter(torch.empty(n_head, self.head_dim))
        self.v_bias = nn.Parameter(torch.empty(d_model))

        anchors = torch.arange(self.head_dim).repeat(n_head, 1)
        free = torch.arange(self.head_dim, d_model).repeat(n_head, 1)
        self.register_buffer("anchor_idx", anchors)
        self.register_buffer("free_idx", free)

    @torch.no_grad()
    def initialize_from_full(
        self, full_weight: torch.Tensor, full_bias: torch.Tensor
    ) -> None:
        q_full, k_full, v_full = full_weight.split(self.d_model, dim=0)
        q_bias_full = full_bias[: self.d_model].view(
            self.n_head, self.head_dim
        )
        v_bias_full = full_bias[2 * self.d_model :]

        candidates = list(
            itertools.combinations(range(self.d_model), self.head_dim)
        )
        eye = torch.eye(
            self.head_dim,
            device=full_weight.device,
            dtype=full_weight.dtype,
        )

        for head in range(self.n_head):
            start = head * self.head_dim
            stop = start + self.head_dim
            q_head = q_full[start:stop]
            k_head = k_full[start:stop]

            anchor_cols = max(
                candidates,
                key=lambda cols: float(
                    torch.linalg.svdvals(q_head[:, list(cols)]).amin()
                ),
            )
            free_cols = tuple(
                index
                for index in range(self.d_model)
                if index not in anchor_cols
            )
            self.anchor_idx[head].copy_(
                torch.tensor(
                    anchor_cols,
                    device=self.anchor_idx.device,
                    dtype=torch.long,
                )
            )
            self.free_idx[head].copy_(
                torch.tensor(
                    free_cols,
                    device=self.free_idx.device,
                    dtype=torch.long,
                )
            )

            anchor = q_head[:, list(anchor_cols)]
            transform = torch.linalg.solve(
                anchor.transpose(0, 1), self.q_scale * eye
            ).transpose(0, 1)
            self.q_free[head].copy_(
                transform @ q_head[:, list(free_cols)]
            )
            self.k_weight[head].copy_(
                torch.linalg.solve(transform.transpose(0, 1), k_head)
            )
            self.q_bias[head].copy_(transform @ q_bias_full[head])

        self.v_weight.copy_(v_full)
        self.v_bias.copy_(v_bias_full)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_heads = []
        k_heads = []
        for head in range(self.n_head):
            anchored = x.index_select(-1, self.anchor_idx[head])
            free = x.index_select(-1, self.free_idx[head])
            q_heads.append(
                self.q_scale * anchored
                + F.linear(free, self.q_free[head], self.q_bias[head])
            )
            k_heads.append(F.linear(x, self.k_weight[head]))

        q = torch.stack(q_heads, dim=2).transpose(1, 2)
        k = torch.stack(k_heads, dim=2).transpose(1, 2)
        v = F.linear(x, self.v_weight, self.v_bias)
        v = v.view(
            x.size(0), x.size(1), self.n_head, self.head_dim
        ).transpose(1, 2)
        return q, k, v


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then remove only the softmax-null
        # key bias while retaining the successful full value bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model :])).clone()
        )
        self.proj = MeanFreeResidualLinear(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.proj = MeanFreeResidualLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        q_bias = self.qkv.bias[:d_model]
        v_bias = self.qkv.bias[d_model:]
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
        qkv = F.linear(x, self.qkv.weight) + full_bias
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        bsz, seqlen, d_model = x.shape
        q, k, v = self.qkv(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, TiedMeanFreeOutput):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedQKV):
            with torch.no_grad():
                full_weight = torch.empty(
                    3 * module.d_model,
                    module.d_model,
                    device=module.v_weight.device,
                    dtype=module.v_weight.dtype,
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                full_bias = torch.zeros(
                    3 * module.d_model,
                    device=module.v_bias.device,
                    dtype=module.v_bias.dtype,
                )
                module.initialize_from_full(full_weight, full_bias)
        elif isinstance(module, TiedMeanFreeOutput):
>>>>>>> REPLACE