MECHANISM: Concentrated highest-band spectral pooling

HYPOTHESIS: Averaging only mel bands 17–19 into one feature will retain at least 85% validation accuracy while reducing recurrent MACs to the prior 18-input level, because it concentrates both removed spectral degrees of freedom in the least-resolved high-frequency region.

INTENDED_EDIT: Reduce the GRU input from 19 to 18 features by preserving mel bands 0–16 individually and averaging the final three bands.

EVIDENCE: The 19-input model pooling bands 18–19 passed at 85.52%, whereas separately pooling bands 16–17 caused the 18-input model to fail at 84.29%; pooling the final three bands tests the same structural reduction without altering the more informative band 16.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(19)
        self.gru = nn.GRU(19, 98, num_layers=1, batch_first=True)
=======
        self.input_norm = nn.LayerNorm(18)
        self.gru = nn.GRU(18, 98, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pooled_frame = torch.cat(
            (frame[:, :18], frame[:, 18:].mean(dim=1, keepdim=True)), dim=1
        )
=======
        pooled_frame = torch.cat(
            (frame[:, :17], frame[:, 17:].mean(dim=1, keepdim=True)), dim=1
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        pooled_frames = torch.cat(
            (frames[:, :, :18], frames[:, :, 18:].mean(dim=2, keepdim=True)), dim=2
        )
=======
        pooled_frames = torch.cat(
            (frames[:, :, :17], frames[:, :, 17:].mean(dim=2, keepdim=True)), dim=2
        )
>>>>>>> REPLACE