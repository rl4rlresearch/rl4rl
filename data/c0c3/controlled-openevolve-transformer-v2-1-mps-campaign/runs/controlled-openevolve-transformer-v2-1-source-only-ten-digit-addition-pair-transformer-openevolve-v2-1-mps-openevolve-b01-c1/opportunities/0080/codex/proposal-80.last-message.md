MECHANISM: Shared-query attention with head-specific memories

HYPOTHESIS: Broadcasting one learned four-dimensional query projection to both attention heads, while retaining head-specific query biases and fully independent key/value projections, will reduce the model from 1,544 to 1,512 parameters while maintaining at least 99% accuracy because both heads can formulate retrievals from a common decoder-state representation yet specialize through their separate learned memories.

INTENDED_EDIT: Replace the two independent query-weight maps with one shared learned query map, preserving separate key/value weights, all query and value biases, causal masking, value bandwidth, initialization variance, and the original RNG sequence.

EVIDENCE: Multi-query attention fell to 6.64% when both keys and values were shared, showing that the combined shared-memory assumption was unsafe but providing no evidence that independent query feature extraction is required. The 1,544-parameter model reaches 99.90%; this patch cleanly tests the opposite decomposition while restoring every head-specific memory pathway implicated by that failure.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class SharedQueryQKV(nn.Linear):
    """One learned query map with independent key/value maps per head."""

    def __init__(self, d_model: int, n_head: int):
        # Consume the same constructor draws as the original full QKV layer.
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        full_weight = self.weight.detach()
        retained_weight = torch.cat(
            (full_weight[: self.head_dim], full_weight[d_model:]), dim=0
        )
        self.out_features = 2 * d_model + self.head_dim
        self.weight = nn.Parameter(retained_weight.clone())
        self.bias = None


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
        self.qkv = SharedQueryQKV(d_model, n_head)
        # Query offsets remain head-specific; only their input-dependent map is
        # shared. Keys remain bias-free and values retain their full bias.
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = MeanFreeResidualLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = self.qkv.bias[:d_model]
        v_bias = self.qkv.bias[d_model:]
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
        qkv = F.linear(x, self.qkv.weight) + full_bias
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = F.linear(x, self.qkv.weight)
        q_shared, k, v = torch.split(
            qkv, (self.head_dim, d_model, d_model), dim=-1
        )

        q = q_shared.unsqueeze(2).expand(
            bsz, seqlen, self.n_head, self.head_dim
        )
        q = q + self.q_bias.view(1, 1, self.n_head, self.head_dim)
        v = v + self.v_bias

        q = q.transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, TwoPrunedInputWeightLinear):
=======
        elif isinstance(module, SharedQueryQKV):
            # Draw the original two query maps so later initialization keeps
            # the same RNG sequence, then combine them at unchanged variance.
            with torch.no_grad():
                full = torch.empty(
                    3 * module.d_model,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                query_heads = full[: module.d_model].view(
                    module.n_head, module.head_dim, module.in_features
                )
                shared_query = query_heads.sum(dim=0) / math.sqrt(module.n_head)
                module.weight.copy_(
                    torch.cat((shared_query, full[module.d_model :]), dim=0)
                )
        elif isinstance(module, TwoPrunedInputWeightLinear):
>>>>>>> REPLACE