"""
Base class for any trainer
"""

import os

import torch
import torch.distributed as dist
import wandb


class BaseTrainer:
    def __init__(
        self,
        train_cfg, logging_cfg, model_cfg,
        global_rank = 0, local_rank = 0, world_size = 1,
        device = None
    ):
        self.rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size

        self.train_cfg = train_cfg
        self.logging_cfg = logging_cfg
        self.model_cfg = model_cfg

        if device is None:
            device = f'cuda:{local_rank}'
        self.device = device
        
        if self.logging_cfg is not None and self.rank == 0:
            log = self.logging_cfg
            wandb.init(
                project=log.project,
                entity=log.name,
                name=log.run_name,
                config={"train": train_cfg, "model": model_cfg},
            )

        if 'cuda' in self.device:
            torch.cuda.set_device(self.local_rank)

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def get_module(self, ema: bool = False):
        if self.world_size == 1:
            if ema:
                return self.ema.ema_model
            else:
                return self.model
        else:
            if ema:
                return self.ema.ema_model.module
            else:
                return self.model.module

    def save(self, save_dict):
        os.makedirs(self.train_cfg.checkpoint_dir, exist_ok=True)

        fp = os.path.join(
            self.train_cfg.checkpoint_dir, f"step_{self.total_step_counter}.pt"
        )

        torch.save(save_dict, fp)

        if 'ema' in save_dict and getattr(self.train_cfg, 'output_path', None) is not None:
            out_d = save_dict['ema']
            prefix = "ema_model.module." if self.world_size > 1 else "ema_model."
            out_d = {k[len(prefix):]: v for k, v in out_d.items() if k.startswith(prefix)}
            os.makedirs(self.train_cfg.output_path, exist_ok = True)
            torch.save(out_d, os.path.join(self.train_cfg.output_path, f"step_{self.total_step_counter}.pt"))

    def get_latest_checkpoint(self, checkpoint_dir: str):
        """
        Find the checkpoint with the largest step number in the given directory.

        :param checkpoint_dir: Directory containing checkpoint files
        :return: Path to the latest checkpoint file
        """
        import glob

        # Get all checkpoint files
        ckpt_pattern = os.path.join(checkpoint_dir, "step_*.pt")
        ckpt_files = glob.glob(ckpt_pattern)

        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        # Extract step numbers and sort
        ckpt_steps = []
        for ckpt_file in ckpt_files:
            # Extract step number from filename like "step_1000.pt"
            basename = os.path.basename(ckpt_file)
            step_str = basename.replace("step_", "").replace(".pt", "")
            try:
                step_num = int(step_str)
                ckpt_steps.append((step_num, ckpt_file))
            except ValueError:
                continue

        if not ckpt_steps:
            raise FileNotFoundError(f"No valid checkpoints found in {checkpoint_dir}")

        # Get checkpoint with largest step number
        ckpt_steps.sort(key=lambda x: x[0])
        latest_path = ckpt_steps[-1][1]

        if self.rank == 0:
            print(f"Loading latest checkpoint: {latest_path} (step {ckpt_steps[-1][0]})")

        return latest_path

    def load(self, path: str, checkpoint_dir: str = None):
        """
        Load a checkpoint from the given path.

        :param path: Path to checkpoint file, or "latest" to load the most recent checkpoint
        :param checkpoint_dir: Directory containing checkpoints (required if path is "latest")
        :return: Loaded checkpoint dictionary
        """
        if path == "latest":
            if checkpoint_dir is None:
                raise ValueError("checkpoint_dir must be provided when path is 'latest'")
            path = self.get_latest_checkpoint(checkpoint_dir)

        return torch.load(path, map_location=self.device, weights_only=False)
