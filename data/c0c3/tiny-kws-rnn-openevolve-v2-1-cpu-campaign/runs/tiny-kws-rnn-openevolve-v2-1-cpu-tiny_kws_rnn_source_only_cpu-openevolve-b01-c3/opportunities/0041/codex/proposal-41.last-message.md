MECHANISM: Candidate-biased gated-state width trim

HYPOTHESIS: A 98-unit state retaining the qualified 31/32/33 reset-update-candidate feedback allocation will achieve at least 85% validation accuracy while reducing total inference MACs from 326,612,880 to approximately 323,313,760.

INTENDED_EDIT: Reduce the recurrent state, gate projections and expansions, temporal summary, and classifier input from 99 to 98 units while preserving the 16-frame schedule and training procedure.

EVIDENCE: The adjacent 99-unit 31/32/33 design qualified at 85.399% accuracy, and the same candidate-biased allocation enabled both the 100- and 99-unit models to qualify; a one-unit state trim is the closest supported structural reduction.

<<<<<<< SEARCH
        self.input_projection = nn.Linear(20, 297)
        self.recurrent_down = nn.Linear(99, 96)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 99),
                nn.Linear(32, 99),
                nn.Linear(33, 99),
            ]
        )
        self.classifier = nn.Linear(198, 8)
=======
        self.input_projection = nn.Linear(20, 294)
        self.recurrent_down = nn.Linear(98, 96)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 98),
                nn.Linear(32, 98),
                nn.Linear(33, 98),
            ]
        )
        self.classifier = nn.Linear(196, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
>>>>>>> REPLACE