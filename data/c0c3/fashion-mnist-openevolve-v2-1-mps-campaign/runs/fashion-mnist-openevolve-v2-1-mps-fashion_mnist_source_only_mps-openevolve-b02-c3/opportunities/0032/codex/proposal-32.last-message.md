MECHANISM: Fine-grained posterior temperature calibration

HYPOTHESIS: Reproducing Reference Design 3 with temperature 0.76 will preserve all 9,315 argmax predictions while lowering validation cross-entropy below 0.1915621.

INTENDED_EDIT: Adopt the verified 7×7 replicate-padded refinement and paired flip supervision, use probability-averaged test-time augmentation, and sharpen its posterior from temperature 0.78 to 0.76.

EVIDENCE: Temperatures 0.90, 0.80, and 0.78 preserved 9,315 correct predictions while successively reducing cross-entropy to 0.1971655, 0.1919746, and 0.1915621; the diminishing improvements indicate an optimum near 0.76.

<<<<<<< SEARCH
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
=======
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False,
            padding_mode="replicate",
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
        logits = torch.zeros(
            (images.shape[0], 10),
            device=images.device,
            dtype=images.dtype,
        )
        for crop, weight in zip(crops, crop_weights):
            logits = logits + weight * self._forward_once(crop)
            logits = logits + weight * self._forward_once(crop.flip(-1))
        return logits / (2.0 * sum(crop_weights))
=======
        probabilities = torch.zeros(
            (images.shape[0], 10),
            device=images.device,
            dtype=images.dtype,
        )
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
        for crop, weight in zip(crops, crop_weights):
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop), dim=1
            )
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
        probabilities = probabilities / (2.0 * sum(crop_weights))
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.76
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    del step, total_steps
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.02,
    )
>>>>>>> REPLACE