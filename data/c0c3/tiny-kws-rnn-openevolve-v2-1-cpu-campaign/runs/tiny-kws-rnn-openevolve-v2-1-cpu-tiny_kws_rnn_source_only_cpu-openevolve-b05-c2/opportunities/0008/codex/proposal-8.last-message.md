MECHANISM: Full-resolution GRU width pruning with dual-timescale readout

HYPOTHESIS: An 80-unit GRU processing all 32 frames and classifying concatenated mean/final states will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 15.8% versus the qualified 88-unit design.

INTENDED_EDIT: Reduce the GRU and summary width from 112 to 80, resize the classifier for a 160-feature mean/final readout, and preserve the complete causal frame schedule and training procedure.

EVIDENCE: The 88-unit full-resolution dual-readout design achieved 86.87% accuracy, and the preceding reduction from 96 to 88 units cost only 0.49 points; this supports testing another eight-unit reduction while avoiding the temporal subsampling that repeatedly failed.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
=======
        self.gru = nn.GRU(20, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(160, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        final_output = hidden[:, 0, :]
        return self.classifier(torch.cat((mean_output, final_output), dim=-1))
>>>>>>> REPLACE