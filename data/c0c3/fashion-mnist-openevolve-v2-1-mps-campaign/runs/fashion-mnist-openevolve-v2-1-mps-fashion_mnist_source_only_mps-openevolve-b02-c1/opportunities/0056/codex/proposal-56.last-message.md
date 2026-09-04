MECHANISM: Tail-consistent BatchNorm state averaging

HYPOTHESIS: Blending BatchNorm running statistics with their sparse tail averages will exceed 9,286 correct predictions by aligning evaluation normalization with the already beneficial 50:50 tail-averaged parameters.

INTENDED_EDIT: Track BatchNorm running means and variances at every existing tail sample and blend them 50:50 with their tail averages at the final step.

EVIDENCE: Half-strength parameter averaging improved validation_correct from 9,280 to 9,286, but the current implementation leaves BatchNorm running statistics at their final-iterate values, creating a normalization mismatch with the blended weights.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=8e-4, weight_decay=2e-4
    )
    optimizer._tail_model = model
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
        with torch.no_grad():
            if not hasattr(optimizer, "_tail_average"):
                optimizer._tail_average = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._tail_average_count = 1
            else:
                optimizer._tail_average_count += 1
                weight = 1.0 / optimizer._tail_average_count
                for average, parameter in zip(
                    optimizer._tail_average, parameters
                ):
                    average.lerp_(parameter.detach(), weight)

            if update == total_steps:
                for parameter, average in zip(
                    parameters, optimizer._tail_average
                ):
                    parameter.lerp_(average, 0.5)
=======
        batch_norm_buffers = [
            buffer
            for module in optimizer._tail_model.modules()
            if isinstance(module, nn.BatchNorm2d)
            for buffer in (module.running_mean, module.running_var)
            if buffer is not None
        ]
        with torch.no_grad():
            if not hasattr(optimizer, "_tail_average"):
                optimizer._tail_average = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._tail_buffer_average = [
                    buffer.detach().clone() for buffer in batch_norm_buffers
                ]
                optimizer._tail_average_count = 1
            else:
                optimizer._tail_average_count += 1
                weight = 1.0 / optimizer._tail_average_count
                for average, parameter in zip(
                    optimizer._tail_average, parameters
                ):
                    average.lerp_(parameter.detach(), weight)
                for average, buffer in zip(
                    optimizer._tail_buffer_average, batch_norm_buffers
                ):
                    average.lerp_(buffer.detach(), weight)

            if update == total_steps:
                for parameter, average in zip(
                    parameters, optimizer._tail_average
                ):
                    parameter.lerp_(average, 0.5)
                for buffer, average in zip(
                    batch_norm_buffers, optimizer._tail_buffer_average
                ):
                    buffer.lerp_(average, 0.5)
>>>>>>> REPLACE