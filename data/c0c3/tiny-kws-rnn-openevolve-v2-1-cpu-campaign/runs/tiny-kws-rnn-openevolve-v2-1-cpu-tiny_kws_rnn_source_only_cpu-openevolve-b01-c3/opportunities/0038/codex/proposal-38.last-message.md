MECHANISM: Cost-neutral feedback-rank reallocation toward candidate generation

HYPOTHESIS: A 100-unit state with 31/32/33 reset-update-candidate feedback widths will achieve at least 85% validation accuracy at approximately 329,912,000 MACs by shifting one coordinate from the less capacity-sensitive reset path to the candidate path.

INTENDED_EDIT: Reduce the state to 100 units and reallocate the fixed 96-dimensional recurrent bottleneck from 32/32/32 to 31/32/33, preserving the 16-frame schedule and training procedure.

EVIDENCE: The 100-unit 32/32/32 model narrowly failed at 84.417%, while the qualified 104-unit 31/32/32 result shows the reset path can tolerate a one-coordinate reduction; the failure of 31/31/31 indicates update/candidate capacity is more valuable, motivating a cost-neutral transfer to the candidate path.

<<<<<<< SEARCH
        self.input_projection = nn.Linear(20, 306)
        self.recurrent_down = nn.Linear(102, 96)
        self.recurrent_up = nn.ModuleList(
            [nn.Linear(32, 102) for _ in range(3)]
        )
        self.classifier = nn.Linear(204, 8)
=======
        self.input_projection = nn.Linear(20, 300)
        self.recurrent_down = nn.Linear(100, 96)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 100),
                nn.Linear(32, 100),
                nn.Linear(33, 100),
            ]
        )
        self.classifier = nn.Linear(200, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 102, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 102, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
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