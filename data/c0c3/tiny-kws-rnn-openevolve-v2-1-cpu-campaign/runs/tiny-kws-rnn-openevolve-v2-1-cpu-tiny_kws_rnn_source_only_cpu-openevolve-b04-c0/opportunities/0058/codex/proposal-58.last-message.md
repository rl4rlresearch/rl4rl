MECHANISM: Earliest-boundary-preserving width–time exchange

HYPOTHESIS: A 69-unit GRU executing 22 steps will retain at least 85% validation accuracy when frame 1 is preserved and frame 2 is removed, while reducing total inference MACs by approximately 1.8% versus the qualifying 68-unit, 23-step model.

INTENDED_EDIT: Increase recurrent and summary width from 68 to 69, resize the classifier from 272 to 276 inputs, and reduce the successful boundary-aware schedule to 22 frames by removing its second-earliest frame instead of its earliest frame.

EVIDENCE: The 68-unit schedule containing frames 1 and 2 achieved 85.52%, whereas the 69-unit variant that removed frame 1 fell to 84.42%; this suggests the earliest boundary frame is disproportionately informative and motivates preserving frame 1 while removing adjacent frame 2.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 68, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(272, 8)
=======
        self.gru = nn.GRU(20, 69, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(276, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 68, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 68), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 69, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 69), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return qualifying_schedule[1:]
=======
        return qualifying_schedule[1:2] + qualifying_schedule[3:]
>>>>>>> REPLACE