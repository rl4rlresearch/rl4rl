MECHANISM: Zero-sum softmax logit parameterization

HYPOTHESIS: A seven-output classifier with the eighth logit derived as the negative sum will retain at least 85% accuracy while reducing exact classifier MACs by 79,055, because it preserves all seven independent softmax decision dimensions.

INTENDED_EDIT: Replace the 97-to-8 classifier with a 97-to-7 classifier and construct an eight-logit zero-sum output without another learned matrix operation.

EVIDENCE: The 97-coordinate readout passed at 85.52%, while reducing its input to 96 failed; exploiting softmax’s redundant common-logit dimension reduces MACs without discarding another recurrent feature.

<<<<<<< SEARCH
        self.classifier = nn.Linear(97, 8)
=======
        self.classifier = nn.Linear(97, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(averaged[:, :97])
=======
        logits = self.classifier(averaged[:, :97])
        return torch.cat((logits, -logits.sum(dim=1, keepdim=True)), dim=1)
>>>>>>> REPLACE