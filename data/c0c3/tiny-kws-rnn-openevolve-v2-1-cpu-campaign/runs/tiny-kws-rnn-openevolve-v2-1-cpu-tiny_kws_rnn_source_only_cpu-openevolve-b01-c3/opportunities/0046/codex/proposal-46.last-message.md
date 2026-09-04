MECHANISM: Max-pooled candidate-biased gated-state width trim

HYPOTHESIS: A 94-unit state with 31/32/33 reset-update-candidate feedback and mean-max-endpoint readout will retain at least 85% validation accuracy while reducing total inference MACs to approximately 310,730,160.

INTENDED_EDIT: Reduce the recurrent state, gate projections and expansions, temporal summaries, and classifier input from 97 to 94 units while preserving the qualified 16-frame schedule and training procedure.

EVIDENCE: The adjacent 95-unit design qualified at 85.521% accuracy and 314,035,800 MACs, while every tested 98-through-95-unit mean-max-endpoint design qualified; a one-unit trim beyond that best available design is the closest untested structural reduction.

<<<<<<< SEARCH
        self.input_projection = nn.Linear(20, 291)
        self.recurrent_down = nn.Linear(97, 96)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 97),
                nn.Linear(32, 97),
                nn.Linear(33, 97),
            ]
        )
        self.classifier = nn.Linear(291, 8)
=======
        self.input_projection = nn.Linear(20, 282)
        self.recurrent_down = nn.Linear(94, 96)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 94),
                nn.Linear(32, 94),
                nn.Linear(33, 94),
            ]
        )
        self.classifier = nn.Linear(282, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
        maximum = torch.full(
            (batch_size, 97), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 94, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 94, device=device, dtype=dtype)
        maximum = torch.full(
            (batch_size, 94), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE