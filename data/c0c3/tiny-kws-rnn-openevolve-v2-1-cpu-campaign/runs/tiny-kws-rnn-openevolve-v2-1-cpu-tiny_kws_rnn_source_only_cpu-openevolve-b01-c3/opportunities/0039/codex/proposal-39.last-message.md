MECHANISM: Candidate-biased feedback with a one-unit state trim

HYPOTHESIS: A 99-unit state with 31/32/33 reset-update-candidate feedback widths will retain at least 85% validation accuracy while reducing total inference MACs to approximately 326,612,880.

INTENDED_EDIT: Reduce the recurrent state and readout from 103 to 99 units, while reallocating the fixed 96-dimensional feedback bottleneck from 32/32/32 to 31/32/33.

EVIDENCE: The qualified 100-unit 31/32/33 design achieved 85.890% accuracy at 329,912,000 MACs, providing the strongest margin among bottleneck-gated designs and making a one-unit state reduction the closest supported cost improvement.

<<<<<<< SEARCH
        self.input_projection = nn.Linear(20, 309)
        self.recurrent_down = nn.Linear(103, 96)
        self.recurrent_up = nn.ModuleList(
            [nn.Linear(32, 103) for _ in range(3)]
        )
        self.classifier = nn.Linear(206, 8)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        reset_code, update_code, candidate_code = torch.tanh(
            self.recurrent_down(hidden)
        ).chunk(3, dim=1)
=======
        reset_code, update_code, candidate_code = torch.split(
            torch.tanh(self.recurrent_down(hidden)), (31, 32, 33), dim=1
        )
>>>>>>> REPLACE