import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        with self._autocast_context():
            outputs = self.model(**batch)
            batch.update(outputs)

            if self.is_train and self.distillation is not None:
                distill_outputs = self.distillation(**batch)
                batch.update(distill_outputs)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            accum_steps = int(self.grad_accum_steps)
            loss_to_backprop = batch["loss"] / accum_steps
            should_step = (self._train_batch_idx + 1) % accum_steps == 0 or (
                self._train_batch_idx + 1
            ) >= self.epoch_len
            if self.use_grad_scaler:
                self.scaler.scale(loss_to_backprop).backward()
                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    self._clip_grad_norm()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                loss_to_backprop.backward()
                if should_step:
                    self._clip_grad_norm()
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            if should_step and self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
