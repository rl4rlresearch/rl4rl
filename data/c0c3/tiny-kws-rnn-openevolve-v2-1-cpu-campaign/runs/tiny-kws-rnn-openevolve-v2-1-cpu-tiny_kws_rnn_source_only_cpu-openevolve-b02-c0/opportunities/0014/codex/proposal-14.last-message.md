MECHANISM: Second adjacent high-frequency mel-band pooling

HYPOTHESIS: Compressing bands 16–17 and 18–19 independently will retain at least 85% accuracy while reducing recurrent MACs by 7,503,705 versus the verified 19-input design.

INTENDED_EDIT: Reduce the GRU input width from 19 to 18 by retaining bands 0–15 and averaging the two highest adjacent band pairs; preserve the verified 99-unit state, 31-step schedule, and classifier.

EVIDENCE: Pooling bands 18–19 achieved 85.52% accuracy, improving on the uncompressed 31-step model’s 85.03%. The previous test of this second pooling change timed out and therefore supplied no contrary accuracy evidence.

<<<<<<< SEARCH
        self.gru = nn.GRU(19, 99, num_layers=1, batch_first=True)
=======
        self.gru = nn.GRU(18, 99, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
        high_band = normalized[..., 18:].mean(dim=-1, keepdim=True)
        return torch.cat((normalized[..., :18], high_band), dim=-1)
=======
        high_bands = normalized[..., 16:].reshape(
            *normalized.shape[:-1], 2, 2
        ).mean(dim=-1)
        return torch.cat((normalized[..., :16], high_bands), dim=-1)
>>>>>>> REPLACE