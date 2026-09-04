MECHANISM: Head-specific learned relative-lag attention

HYPOTHESIS: Replacing the 184-parameter absolute-position table with 44 learned head-specific causal lag biases will retain at least 99% accuracy while reducing the model from 1587 to 1447 parameters, because the two attention heads can directly specialize to the previous-output and aligned-input offsets without consuming residual-stream dimensions.

INTENDED_EDIT: Remove absolute positional vectors, add a learned relative-lag bias to each attention head, fix the softmax-redundant zero-lag coordinate, and preserve the verified initialization random stream.

EVIDENCE: Fixed Fourier positions achieved only 14.4% accuracy, showing that generic positional features were inadequate, while width-preserving quotient changes consistently exceeded 99%. Learned lag biases test a different mechanism: task-relevant positional geometry is learned directly in attention logits while all eight content channels remain available.

<<<<<<< SEARCH
        self.proj = OneColumnShiftGaugeLinear(d_model, d_model)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
=======
        self.proj = OneColumnShiftGaugeLinear(d_model, d_model)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
        # A constant shift of every lag bias cancels in the softmax, so lag
        # zero is the fixed reference coordinate.
        self.relative_bias_coordinates = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 1)
        )
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
=======
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seqlen, device=x.device)
        lags = (positions[:, None] - positions[None, :]).clamp_min(0)
        relative_bias = F.pad(self.relative_bias_coordinates, (1, 0))
        att = att + relative_bias[:, lags].unsqueeze(0)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = ShiftGaugeEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.apply(self._init_weights)
=======
        self.cfg = cfg
        self.token_emb = ShiftGaugeEmbedding(cfg.vocab_size, cfg.d_model)

        # Preserve the constructor-time random draw of the removed Embedding
        # so all downstream modules retain their verified initialization.
        conceptual_position = torch.empty(cfg.max_seq_len, cfg.d_model)
        nn.init.normal_(conceptual_position)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Reproduce the original apply-time order, including the removed
        # position table's draw, before initializing the transformer block.
        self.token_emb.reset_parameters()
        nn.init.normal_(conceptual_position, mean=0.0, std=0.02)
        self.blocks.apply(self._init_weights)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        x = self.drop(self.token_emb(idx))
>>>>>>> REPLACE