MECHANISM: Width–time rebalancing with variance-assisted readout

HYPOTHESIS: A 65-unit GRU with deviation readout processing 26 uniformly spaced frames will retain at least 85% validation accuracy while using approximately 1.0% fewer dense inference MACs than the verified 64-unit, 27-step model.

INTENDED_EDIT: Increase the recurrent and summary width from 64 to 65 units, resize the classifier input from 256 to 260 features, and reduce the schedule from 27 to 26 steps.

EVIDENCE: The deviation readout gave the 65-unit, 27-step model 86.50% accuracy, while reducing both width and steps to 64 units and 26 frames reached 84.05%; restoring one unit preserves an estimated MAC advantage over the qualifying 64-unit, 27-step design while testing whether added recurrent capacity recovers the threshold.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)
=======
        self.gru = nn.GRU(20, 65, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(260, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 64), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 65, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 65, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 65, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 65), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        target_steps = min(27, available_frames)
=======
        target_steps = min(26, available_frames)
>>>>>>> REPLACE