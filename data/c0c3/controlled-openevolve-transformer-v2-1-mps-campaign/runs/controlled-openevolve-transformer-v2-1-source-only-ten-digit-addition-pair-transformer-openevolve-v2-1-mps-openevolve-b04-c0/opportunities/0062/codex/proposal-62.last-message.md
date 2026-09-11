MECHANISM: Query-bias softmax-gauge anchoring

HYPOTHESIS: Fixing `q_bias[0]` at zero will reduce the model from 1596 to 1595 parameters while retaining at least 99% accuracy, because one query-shift direction is invisible to attention softmax and the remaining seven query-bias coordinates preserve trainable content-independent attention control.

INTENDED_EDIT: Replace the eight-parameter query bias with seven learned coordinates, reconstructing coordinate 0 as a fixed zero while preserving the existing zero initialization.

EVIDENCE: The 1596-parameter design achieved 99.94%, while removing the sole remaining `fc1.bias[8]` previously collapsed accuracy to 77.57%; this motivates leaving that initialization-sensitive parameter intact and testing an untouched attention-side redundancy instead.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 1))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias = torch.cat((self.q_bias_rest.new_zeros(1), self.q_bias_rest))
        q = q + q_bias
        v = v + self.v_bias
>>>>>>> REPLACE