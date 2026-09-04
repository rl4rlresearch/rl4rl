MECHANISM: Width–time MAC rebalancing

HYPOTHESIS: A 67-unit GRU processing 26 uniformly spaced frames will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.1% versus the verified 66-unit, 27-step model.

INTENDED_EDIT: Increase recurrent and summary width from 66 to 67 units, resize the classifier input from 198 to 201 features, and reduce the uniform full-utterance schedule from 27 to 26 steps.

EVIDENCE: The 66-unit, 27-step model achieved 85.77% accuracy and improved MACs over the qualifying 65-unit, 28-step model; continuing the same one-unit-for-one-step exchange tests whether modest added recurrent capacity can preserve accuracy at a lower net cost.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 66, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(198, 8)
=======
        self.gru = nn.GRU(20, 67, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(201, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 66, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 66, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 66), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 67, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 67), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        target_steps = min(27, available_frames)
=======
        target_steps = min(26, available_frames)
>>>>>>> REPLACE