MECHANISM: Initialization-preserving elimination of softmax-invariant key bias

HYPOTHESIS: Removing all eight key-bias parameters while preserving the successful model’s RNG consumption and shared QKV weight initialization will produce a 1,635-parameter model with at least 99% accuracy.

INTENDED_EDIT: Retain the combined QKV projection, discard its registered bias after construction, and learn only query and value biases while inserting a fixed-zero key bias.

EVIDENCE: The earlier 1,636-parameter key-bias removal changed the projection structure and achieved only 81.87%, while the current anchored model achieved 99.95%; this patch tests the same exact attention redundancy without perturbing initialization of the functionally relevant weights.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        # Construct with a full bias so downstream modules consume the same
        # initialization RNG sequence, then retain only the useful Q/V biases.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        qkv = self.qkv(x)
        q_bias, v_bias = self.qv_bias.chunk(2)
        qkv = qkv + torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE