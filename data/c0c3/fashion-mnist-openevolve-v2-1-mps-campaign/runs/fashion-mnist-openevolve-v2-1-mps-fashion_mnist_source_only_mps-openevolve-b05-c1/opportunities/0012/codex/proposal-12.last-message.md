MECHANISM: Identity-centered squeeze-and-excitation residual recalibration

HYPOTHESIS: Adding lightweight channel-attention gates to every residual block will exceed 9,204 correct predictions by improving feature selection while preserving the verified architecture, augmentation, and optimization behavior.

INTENDED_EDIT: Add identity-initialized squeeze-and-excitation gates to all residual blocks, increasing learned parameters from 245,044 to 247,528 without changing exposure, batch size, or inference views.

EVIDENCE: The current topology reaches 9,204 correct and leaves 4,956 parameters unused, while the prior large architectural redesign could not be verified; a 2,484-parameter, topology-preserving attention addition is a lower-risk test of additional representational capacity.

<<<<<<< SEARCH
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels == out_channels:
=======
        self.bn2 = nn.BatchNorm2d(out_channels)
        attention_channels = max(4, out_channels // 8)
        self.attention_reduce = nn.Linear(out_channels, attention_channels)
        self.attention_expand = nn.Linear(attention_channels, out_channels)
        nn.init.zeros_(self.attention_expand.weight)
        nn.init.zeros_(self.attention_expand.bias)
        if in_channels == out_channels:
>>>>>>> REPLACE

<<<<<<< SEARCH
        outputs = self.bn2(self.conv2(outputs))
        return F.gelu(outputs + residual)
=======
        outputs = self.bn2(self.conv2(outputs))
        attention = F.adaptive_avg_pool2d(outputs, 1).flatten(1)
        attention = F.gelu(self.attention_reduce(attention))
        attention = 2.0 * torch.sigmoid(self.attention_expand(attention))
        outputs = outputs * attention[:, :, None, None]
        return F.gelu(outputs + residual)
>>>>>>> REPLACE