MECHANISM: AdamW-controlled headwise attention-branch calibration

HYPOTHESIS: Adding zero-initialized per-head biases to the four output gates will lower val_bpb below 0.991177 while retaining at least 470M trained tokens.

INTENDED_EDIT: Give each output gate an input-independent headwise scale parameter and optimize its bias alongside its weight with the best-verified 0.0115 AdamW learning rate.

EVIDENCE: Moving only output-gate weights to AdamW at 0.0115 achieved the best val_bpb, 0.991177, while moving all gate matrices to AdamW regressed; this motivates expanding adaptive control specifically within the successful output gates.

<<<<<<< SEARCH
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False) if is_gated_layer else None
=======
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=True) if is_gated_layer else None
>>>>>>> REPLACE

<<<<<<< SEARCH
            if block.attn.out_gate is not None:
                torch.nn.init.zeros_(block.attn.out_gate.weight)
=======
            if block.attn.out_gate is not None:
                torch.nn.init.zeros_(block.attn.out_gate.weight)
                torch.nn.init.zeros_(block.attn.out_gate.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        output_gate_params = [
            block.attn.out_gate.weight for block in self.transformer.h
            if block.attn.out_gate is not None
        ]
=======
        output_gate_params = [
            p for block in self.transformer.h
            if block.attn.out_gate is not None
            for p in block.attn.out_gate.parameters()
        ]
>>>>>>> REPLACE