MECHANISM: Max-supported recurrent width compression

HYPOTHESIS: An 85-unit GRU with the successful 26-frame mean-max-terminal readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.1%, from 581.19M to 569.02M.

INTENDED_EDIT: Reduce the GRU and summary width from 86 to 85 and resize the concatenated classifier input from 258 to 255, preserving the verified frame schedule and training procedure.

EVIDENCE: The 86-unit, 26-step model reached 86.01% after max pooling rescued the same schedule’s 84.91% mean-terminal result, while an 85-unit model previously reached 86.63% at 27 steps; this supports testing the smallest width reduction without removing the accuracy-restoring readout.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(258, 8)
=======
        self.gru = nn.GRU(20, 85, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(255, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 86, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 85, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 85, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 85, device=device, dtype=dtype)
>>>>>>> REPLACE