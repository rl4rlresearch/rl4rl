MECHANISM: Width–time rebalancing with deviation-assisted readout

HYPOTHESIS: A 67-unit GRU processing 24 uniformly spaced frames will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.4% versus the verified 66-unit, 25-step model.

INTENDED_EDIT: Increase recurrent and summary width from 66 to 67 units, resize the classifier input from 264 to 268 features, and reduce the uniform schedule from 25 to 24 steps.

EVIDENCE: Exchanging one step for one hidden unit previously produced qualifying 65-unit/26-step and 66-unit/25-step deviation-readout models; the latest achieved 85.64% accuracy, motivating the next incremental exchange.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 66, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(264, 8)
=======
        self.gru = nn.GRU(20, 67, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(268, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 66, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 66, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 66, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 66), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 67, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 67), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        target_steps = min(25, available_frames)
=======
        target_steps = min(24, available_frames)
>>>>>>> REPLACE