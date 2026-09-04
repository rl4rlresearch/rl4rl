MECHANISM: Recurrent-to-readout capacity reallocation

HYPOTHESIS: Restoring the informative final-state view while reducing the paired GRU from 69 to 68 units will retain at least 85% validation accuracy and reduce total inference MACs from 240,235,920 to approximately 234,980,800.

INTENDED_EDIT: Resize the paired GRU to 68 units and classify four 68-dimensional views: early mean, late mean, temporal maximum, and final recurrent state.

EVIDENCE: The 69-unit four-view model achieved 85.89% accuracy, substantially above the threshold; removing its final-state view reduced accuracy to 85.28%, showing that cheap readout capacity can support recurrent-width pruning.

<<<<<<< SEARCH
        self.gru = nn.GRU(40, 69, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(207, 7)
=======
        self.gru = nn.GRU(40, 68, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(272, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 69, device=device, dtype=dtype)
        early_summary = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        late_summary = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 69, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 68, device=device, dtype=dtype)
        early_summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        late_summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 68, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
                early_summary / early_count,
                late_summary / late_count,
                maximum,
=======
                early_summary / early_count,
                late_summary / late_count,
                maximum,
                hidden[:, 0, :],
>>>>>>> REPLACE