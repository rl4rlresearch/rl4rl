MECHANISM: Antithetic flip-paired supervision

HYPOTHESIS: Supervising both horizontal orientations of every exposed image—especially during the clean second half—will better align training with ten-view inference and exceed 9,142 correct validation predictions.

INTENDED_EDIT: Concatenate each training batch with its horizontal mirror and compute the existing smoothed cross-entropy over both views in one forward pass.

EVIDENCE: The current design reached 9,142 correct with flip-averaged inference, but flip augmentation stops halfway through training; maintaining explicit flip supervision throughout directly reinforces the successful inference symmetry.

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