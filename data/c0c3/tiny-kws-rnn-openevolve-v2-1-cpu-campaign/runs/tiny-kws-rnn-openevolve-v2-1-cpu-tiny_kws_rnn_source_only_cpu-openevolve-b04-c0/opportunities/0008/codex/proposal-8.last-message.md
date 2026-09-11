MECHANISM: Width-for-temporal-coverage rebalancing

HYPOTHESIS: A 92-unit GRU processing all 32 frames with the dual mean/final-state readout will retain at least 85% accuracy while reducing exact MACs versus the verified 96-unit, 31-step model.

INTENDED_EDIT: Reduce recurrent width from 96 to 92 and restore frame 0, trading one additional step for a structurally cheaper recurrent matrix at every step.

EVIDENCE: The 104-unit dual-readout model reached 87.73%, showing the readout benefits from near-full temporal coverage; the 96-unit model still met 85.03%, while prior single-frame pruning caused a measurable accuracy loss. Since 92 units over 32 steps require fewer recurrent MACs than 96 units over 31 steps, restoring the omitted frame may offset the smaller state without sacrificing the primary cost objective.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)
=======
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(184, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return list(range(1, available_frames))
=======
        return list(range(available_frames))
>>>>>>> REPLACE