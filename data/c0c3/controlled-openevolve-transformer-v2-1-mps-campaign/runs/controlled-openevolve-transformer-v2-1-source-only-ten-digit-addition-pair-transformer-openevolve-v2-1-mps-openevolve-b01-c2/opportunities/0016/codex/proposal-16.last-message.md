MECHANISM: Single-coordinate query-bias head-basis gauge fixing

HYPOTHESIS: Combining the verified distributed LayerNorm anchors with one fixed-zero query-bias coordinate will yield a 1,609-parameter model with at least 99% accuracy, because an invertible query/key basis change within the affected attention head can eliminate that bias coordinate without changing attention logits.

INTENDED_EDIT: Anchor one scale coordinate in `ln1` as in the qualified 1,610-parameter design, then retain seven learned query-bias coordinates and synthesize the eighth as zero.

EVIDENCE: Distributed LayerNorm anchoring achieved 99.97% with 1,610 parameters. The failed 1,609-parameter `fc2` bias anchor targeted the residual pathway; this instead removes an independently redundant attention-head basis coordinate while preserving its zero-initialized value.

<<<<<<< SEARCH
        # Preserve fused projection construction while retaining only query bias.
        # Key bias cancels in softmax and value bias is absorbed by proj.bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
=======
        # Retain all but one query-bias coordinate. A query/key basis change
        # within the first head can rotate the query bias away from that axis.
        # Key bias cancels in softmax and value bias is absorbed by proj.bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = self.qkv.bias
        fused_bias = torch.cat(
            (q_bias, q_bias.new_zeros(d_model), q_bias.new_zeros(d_model))
        )
=======
        learned_q_bias = self.qkv.bias
        q_bias = torch.cat(
            (learned_q_bias.new_zeros(1), learned_q_bias)
        )
        fused_bias = torch.cat(
            (q_bias, q_bias.new_zeros(d_model), q_bias.new_zeros(d_model))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln1 = AnchoredScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE