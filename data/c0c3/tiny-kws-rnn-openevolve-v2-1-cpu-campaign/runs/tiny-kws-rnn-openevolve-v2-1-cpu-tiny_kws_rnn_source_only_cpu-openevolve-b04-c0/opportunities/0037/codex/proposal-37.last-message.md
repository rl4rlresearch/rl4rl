MECHANISM: Width–time MAC rebalancing

HYPOTHESIS: A 66-unit GRU processing 27 uniformly spaced frames will recover validation accuracy to at least 85% while reducing total dense inference MACs by approximately 0.9% versus the verified 65-unit, 28-step model.

INTENDED_EDIT: Increase recurrent width from 65 to 66 units, resize the classifier accordingly, and reduce the frame schedule from 28 to 27 steps.

EVIDENCE: The 65-unit model qualified at 28 steps with 85.77% accuracy but narrowly failed at 27 steps with 84.79%; the verified 66-unit model achieved 86.50% at 32 steps, suggesting one extra hidden unit may restore the small deficit while the removed recurrent step still yields a net MAC reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 65, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(195, 8)
=======
        self.gru = nn.GRU(20, 66, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(198, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 65, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 65, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 65), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 66, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 66, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 66), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        target_steps = min(28, available_frames)
=======
        target_steps = min(27, available_frames)
>>>>>>> REPLACE