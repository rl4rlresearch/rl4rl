MECHANISM: Width–time rebalancing with deviation-assisted readout

HYPOTHESIS: A 66-unit GRU with deviation readout processing 25 uniformly spaced frames will retain at least 85% validation accuracy while reducing dense inference MACs by approximately 1.2% versus the verified 65-unit, 26-step model.

INTENDED_EDIT: Increase recurrent and summary width from 65 to 66 units, resize the classifier input from 260 to 264 features, and reduce the uniform schedule from 26 to 25 steps.

EVIDENCE: The 65-unit deviation-readout model achieved 86.13% at 26 steps, providing accuracy margin, while the same width–time exchange previously allowed 66 units at 27 steps to outperform the qualifying 65-unit, 28-step design in MACs.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 65, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(260, 8)
=======
        self.gru = nn.GRU(20, 66, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(264, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 65, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 65, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 65, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 65), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 66, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 66, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 66, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 66), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        target_steps = min(26, available_frames)
=======
        target_steps = min(25, available_frames)
>>>>>>> REPLACE