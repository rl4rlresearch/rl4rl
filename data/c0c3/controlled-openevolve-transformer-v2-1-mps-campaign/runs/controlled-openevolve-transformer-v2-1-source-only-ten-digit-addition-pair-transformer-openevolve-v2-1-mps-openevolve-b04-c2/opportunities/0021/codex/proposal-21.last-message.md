MECHANISM: Global tied-embedding gauge quotient with verified tail positional compaction

HYPOTHESIS: Quotienting the single global common mode of the tied token-embedding matrix on top of the verified final-four positional design will produce a 1,628-parameter model with accuracy at least 99%, because this mode only adds LayerNorm-canceled input shifts and a softmax-invariant common logit shift.

INTENDED_EDIT: Extend positional compaction to the verified final four rows, represent the tied token matrix in an implicit orthonormal zero-sum basis, and preserve its virtual full-coordinate AdamW dynamics.

EVIDENCE: The final-four positional design achieved 99.81% at 1,629 parameters, while extending to a fifth tail row fell to 96.93% and removing an `ln2` coordinate fell to 93.29%; this tests a different exact one-dimensional symmetry without further compressing sensitive positional or LayerNorm parameters.

<<<<<<< SEARCH
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class CausalSelfAttention(nn.Module):
=======
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


def zero_sum_project(full: torch.Tensor) -> torch.Tensor:
    """Project the final axis into an implicit orthonormal zero-sum basis."""
    count = full.size(-1)
    index = torch.arange(1, count, device=full.device, dtype=full.dtype)
    scale = torch.sqrt(index * (index + 1.0))
    prefix = full[..., :-1].cumsum(dim=-1)
    return (prefix - index * full[..., 1:]) / scale


def zero_sum_expand(compact: torch.Tensor) -> torch.Tensor:
    """Expand implicit orthonormal zero-sum coordinates along the final axis."""
    count = compact.size(-1) + 1
    index = torch.arange(1, count, device=compact.device, dtype=compact.dtype)
    scale = torch.sqrt(index * (index + 1.0))
    positive = compact / scale
    positive = torch.flip(
        torch.cumsum(torch.flip(positive, dims=(-1,)), dim=-1),
        dims=(-1,),
    )
    positive = torch.cat(
        (positive, torch.zeros_like(compact[..., :1])), dim=-1
    )
    negative = torch.cat(
        (
            torch.zeros_like(compact[..., :1]),
            -compact * torch.sqrt(index / (index + 1.0)),
        ),
        dim=-1,
    )
    return positive + negative


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final two positions, which have the shortest causal influence.
        self.compact_pos_count = 4
=======
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final four positions, as in the verified 1,629-parameter design.
        self.compact_pos_count = 6
>>>>>>> REPLACE

<<<<<<< SEARCH
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-2:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-2].reshape(-1),
                )
            )
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
=======
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-4:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-4].reshape(-1),
                )
            )
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)

        # Adding one scalar to every coordinate of every tied token embedding
        # is invisible: pre-norm paths discard the input shift and the output
        # receives only a common logit shift. Remove that global gauge mode.
        with torch.no_grad():
            compact_token = zero_sum_project(
                self.token_emb.weight.detach().reshape(-1)
            )
        self.token_emb.weight = nn.Parameter(compact_token)
        self.lm_head.weight = self.token_emb.weight
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + F.embedding(pos, pos_weight)
=======
        token_weight = zero_sum_expand(self.token_emb.weight).view(
            self.cfg.vocab_size, self.cfg.d_model
        )
        x = F.embedding(idx, token_weight) + F.embedding(pos, pos_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
=======
        logits = F.linear(x, token_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
from src.model import ModelConfig, TinyDecoderLM, count_parameters
=======
from src.model import (
    ModelConfig,
    TinyDecoderLM,
    count_parameters,
    zero_sum_expand,
    zero_sum_project,
)
>>>>>>> REPLACE

<<<<<<< SEARCH
    mlp_gauge_params = [(blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks]
    pos_param = model.pos_emb.weight
    pos_basis = model.pos_basis
    compact_pos_count = model.compact_pos_count
    pos_compact_size = compact_pos_count * (model_cfg.d_model - 1)
    gauge_ids = {id(param) for param, _ in mlp_gauge_params}
    gauge_ids.add(id(pos_param))
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if id(param) not in gauge_ids],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    mlp_gauge_states = [
        {
            "step": 0,
            "exp_avg": basis.new_zeros(basis.size(0)),
            "exp_avg_sq": basis.new_zeros(basis.size(0)),
        }
        for _, basis in mlp_gauge_params
    ]
    pos_gauge_state = {
        "step": 0,
        "exp_avg": pos_basis.new_zeros(model_cfg.max_seq_len * model_cfg.d_model),
        "exp_avg_sq": pos_basis.new_zeros(model_cfg.max_seq_len * model_cfg.d_model),
    }
=======
    mlp_gauge_params = [(blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks]
    pos_param = model.pos_emb.weight
    pos_basis = model.pos_basis
    compact_pos_count = model.compact_pos_count
    pos_compact_size = compact_pos_count * (model_cfg.d_model - 1)
    token_param = model.token_emb.weight
    token_full_size = model_cfg.vocab_size * model_cfg.d_model
    gauge_ids = {id(param) for param, _ in mlp_gauge_params}
    gauge_ids.add(id(pos_param))
    gauge_ids.add(id(token_param))
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if id(param) not in gauge_ids],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    mlp_gauge_states = [
        {
            "step": 0,
            "exp_avg": basis.new_zeros(basis.size(0)),
            "exp_avg_sq": basis.new_zeros(basis.size(0)),
        }
        for _, basis in mlp_gauge_params
    ]
    pos_gauge_state = {
        "step": 0,
        "exp_avg": pos_basis.new_zeros(model_cfg.max_seq_len * model_cfg.d_model),
        "exp_avg_sq": pos_basis.new_zeros(model_cfg.max_seq_len * model_cfg.d_model),
    }
    token_gauge_state = {
        "step": 0,
        "exp_avg": token_param.new_zeros(token_full_size),
        "exp_avg_sq": token_param.new_zeros(token_full_size),
    }
>>>>>>> REPLACE

<<<<<<< SEARCH
                param.mul_(1.0 - lr_now * train_cfg.weight_decay)
                param.add_(quotient_update, alpha=-lr_now / bias_correction1)

            if pos_param.grad is not None:
=======
                param.mul_(1.0 - lr_now * train_cfg.weight_decay)
                param.add_(quotient_update, alpha=-lr_now / bias_correction1)

            if token_param.grad is not None:
                state = token_gauge_state
                state["step"] += 1
                full_grad = zero_sum_expand(token_param.grad)
                state["exp_avg"].lerp_(full_grad, 1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    full_grad, full_grad, value=1.0 - beta2
                )
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denom = (
                    state["exp_avg_sq"].sqrt()
                    .div_(math.sqrt(bias_correction2))
                    .add_(eps)
                )
                compact_update = zero_sum_project(state["exp_avg"] / denom)
                token_param.mul_(1.0 - lr_now * train_cfg.weight_decay)
                token_param.add_(
                    compact_update, alpha=-lr_now / bias_correction1
                )

            if pos_param.grad is not None:
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:2], full_update[-2:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-2].reshape(-1),
                    )
                )
=======
                selected_update = torch.cat(
                    (full_update[:2], full_update[-4:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-4].reshape(-1),
                    )
                )
>>>>>>> REPLACE