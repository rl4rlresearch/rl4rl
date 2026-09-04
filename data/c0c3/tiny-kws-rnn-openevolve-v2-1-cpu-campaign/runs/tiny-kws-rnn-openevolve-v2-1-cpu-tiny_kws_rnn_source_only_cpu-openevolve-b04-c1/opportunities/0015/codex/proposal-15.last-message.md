MECHANISM: Width reduction with salience-preserving temporal readout

HYPOTHESIS: An 80-unit GRU using mean, maximum, and final recurrent outputs will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 8.4% versus the verified 84-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 84 to 80 and resize all recurrent state tensors and the three-way classifier input accordingly, while preserving all 32 causal frames and the successful max-augmented readout.

EVIDENCE: The 84-unit mean-plus-final model missed the threshold at 84.79%, but adding the maximum recurrent output raised accuracy to 86.75% with the same recurrent core; that 1.75-point margin motivates testing a four-unit structural reduction without discarding temporal information.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 8)
=======
        self.gru = nn.GRU(20, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(240, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 84, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 80, device=device, dtype=dtype)
>>>>>>> REPLACE