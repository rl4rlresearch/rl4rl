MECHANISM: Conservative dual-readout width reduction

HYPOTHESIS: An 88-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing exact dense inference MACs relative to the verified 92-unit model.

INTENDED_EDIT: Reduce the recurrent hidden state and temporal summary from 92 to 88 units, and reduce the dual-readout classifier input from 184 to 176 features.

EVIDENCE: The 92-unit, 32-step dual-readout model achieved 87.12% accuracy, 2.12 points above the requirement, while already outperforming the 96-unit, 31-step model in both accuracy and MACs; this supports testing a modest structural width reduction without discarding temporal information.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(184, 8)
=======
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
>>>>>>> REPLACE