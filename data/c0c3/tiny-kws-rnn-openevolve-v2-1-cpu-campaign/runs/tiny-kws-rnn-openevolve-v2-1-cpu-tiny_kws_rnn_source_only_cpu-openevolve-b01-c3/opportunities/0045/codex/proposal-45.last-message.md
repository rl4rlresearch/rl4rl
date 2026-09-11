MECHANISM: Max-pooled candidate-biased gated-state width trim

HYPOTHESIS: A 95-unit state retaining the 31/32/33 feedback allocation and mean-max-endpoint readout will achieve at least 85% validation accuracy while reducing total inference MACs from 317,341,440 to approximately 314,035,800.

INTENDED_EDIT: Reduce the recurrent state, gate projections and expansions, temporal summaries, and classifier input from 96 to 95 units while preserving the 16-frame schedule and training procedure.

EVIDENCE: The adjacent 96-unit design qualified at 85.276% and 317,341,440 MACs, while the 98-, 97-, and 96-unit versions all qualified; the one-unit trim is the closest untested structural reduction and should save exactly 3,305,640 MACs.

<<<<<<< SEARCH
        self.input_projection = nn.Linear(20, 288)
        self.recurrent_down = nn.Linear(96, 96)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 96),
                nn.Linear(32, 96),
                nn.Linear(33, 96),
            ]
        )
        self.classifier = nn.Linear(288, 8)
=======
        self.input_projection = nn.Linear(20, 285)
        self.recurrent_down = nn.Linear(95, 96)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 95),
                nn.Linear(32, 95),
                nn.Linear(33, 95),
            ]
        )
        self.classifier = nn.Linear(285, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
        maximum = torch.full(
            (batch_size, 96), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 95, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 95, device=device, dtype=dtype)
        maximum = torch.full(
            (batch_size, 95), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE