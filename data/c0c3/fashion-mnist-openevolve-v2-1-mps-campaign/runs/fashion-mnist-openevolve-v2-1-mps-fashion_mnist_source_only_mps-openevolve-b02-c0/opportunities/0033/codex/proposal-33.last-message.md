MECHANISM: BatchNorm-aligned late-checkpoint averaging

HYPOTHESIS: Averaging BatchNorm running statistics alongside the successful late parameter snapshots will exceed 9,252 correct predictions by evaluating the averaged weights with activation statistics from the same trajectory window.

INTENDED_EDIT: Retain the model on the optimizer and jointly average/copy BatchNorm running means and variances at every existing late checkpoint.

EVIDENCE: Sparse late-checkpoint parameter averaging improved validation_correct from 9,249 to 9,252, but the current implementation leaves BatchNorm statistics at their final, unaveraged values; aligning those statistics directly refines the only recently successful mechanism.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=6.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=6.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )
    optimizer._averaging_model = model
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
        average_count = getattr(optimizer, "_late_average_count", 0)
        with torch.no_grad():
            if average_count == 0:
                optimizer._late_averaged_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
            else:
                update_weight = 1.0 / (average_count + 1)
                for average, parameter in zip(
                    optimizer._late_averaged_parameters, parameters
                ):
                    average.lerp_(parameter.detach(), update_weight)
            optimizer._late_average_count = average_count + 1

            if is_final_step:
                for parameter, average in zip(
                    parameters, optimizer._late_averaged_parameters
                ):
                    parameter.copy_(average)
=======
        batch_norm_buffers = [
            buffer
            for module in optimizer._averaging_model.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            for buffer in (module.running_mean, module.running_var)
            if buffer is not None
        ]
        average_count = getattr(optimizer, "_late_average_count", 0)
        with torch.no_grad():
            if average_count == 0:
                optimizer._late_averaged_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._late_averaged_batch_norm_buffers = [
                    buffer.detach().clone() for buffer in batch_norm_buffers
                ]
            else:
                update_weight = 1.0 / (average_count + 1)
                for average, parameter in zip(
                    optimizer._late_averaged_parameters, parameters
                ):
                    average.lerp_(parameter.detach(), update_weight)
                for average, buffer in zip(
                    optimizer._late_averaged_batch_norm_buffers,
                    batch_norm_buffers,
                ):
                    average.lerp_(buffer.detach(), update_weight)
            optimizer._late_average_count = average_count + 1

            if is_final_step:
                for parameter, average in zip(
                    parameters, optimizer._late_averaged_parameters
                ):
                    parameter.copy_(average)
                for buffer, average in zip(
                    batch_norm_buffers,
                    optimizer._late_averaged_batch_norm_buffers,
                ):
                    buffer.copy_(average)
>>>>>>> REPLACE