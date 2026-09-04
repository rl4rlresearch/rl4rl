MECHANISM: Second selective reset-gate bottleneck trim

HYPOTHESIS: Reducing only the reset feedback path from 31 to 30 coordinates will retain at least 85% validation accuracy while reducing total inference MACs from 340,396,160 to approximately 337,683,840.

INTENDED_EDIT: Resize the shared recurrent projection from 95 to 94 outputs, split it into 30/32/32 reset-update-candidate codes, and resize only the reset expansion layer.

EVIDENCE: The selective 31/32/32 design qualified at 85.153% and outperformed the failed 31/31/31 design, indicating update and candidate capacity should remain intact while the reset path is the most promising target for another minimal structural reduction.

<<<<<<< SEARCH
        self.recurrent_down = nn.Linear(104, 95)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 104),
                nn.Linear(32, 104),
                nn.Linear(32, 104),
            ]
        )
=======
        self.recurrent_down = nn.Linear(104, 94)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(30, 104),
                nn.Linear(32, 104),
                nn.Linear(32, 104),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        reset_code, update_code, candidate_code = torch.split(
            torch.tanh(self.recurrent_down(hidden)), (31, 32, 32), dim=1
        )
=======
        reset_code, update_code, candidate_code = torch.split(
            torch.tanh(self.recurrent_down(hidden)), (30, 32, 32), dim=1
        )
>>>>>>> REPLACE