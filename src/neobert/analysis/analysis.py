import torch
from typing import Union
from torch.nn import CrossEntropyLoss

# from NeoBERT.NeoBERT_dev.src.neobert.pretraining import metrics
# from NeoBERT.NeoBERT_dev.src.neobert import dataloader
from omegaconf import DictConfig
from collections import defaultdict
import pandas as pd
from ..tokenizer import get_tokenizer
from .analysis_utils import AnalysisMetrics
import wandb
import matplotlib.pyplot as plt

# Add a small legend at the bottom of the figure
import matplotlib.lines as mlines
from matplotlib.legend import Legend
import numpy as np


class AnalysisLogger(AnalysisMetrics):
    def __init__(
        self,
        cfg,
        accelerator,
        buffer_size_finetuning_token=5000,
        buffer_size_finetuning_seq=500,
        buffer_size_seq=1000,
        buffer_size_token=5000,
    ):
        self.cfg = cfg
        self.tokenizer = get_tokenizer(**cfg.tokenizer)
        self.accelerator = accelerator

        # Initialise dictionaries to store expert usage per token and per sequence
        self.token_expert_usage_dict = defaultdict(lambda: [])
        self.sequence_expert_usage_dict = defaultdict(lambda: [])

        # Initialise buffers for Correlation between CE and expert usage
        self.ce_loss_seq_buffer = []
        self.expert_usage_seq_buffer = []
        self.buffer_size_seq = buffer_size_seq
        self.ce_loss_token_buffer = []
        self.expert_usage_token_buffer = []
        self.buffer_size_token = buffer_size_token
        self.expert_usage_seq_buffer_finetuning = []
        self.task_loss_seq_buffer_finetuning = []
        self.expert_usage_token_buffer_finetuning = []
        self.task_loss_token_buffer_finetuning = []
        self.buffer_size_finetuning_seq = buffer_size_finetuning_seq
        self.buffer_size_finetuning_token = buffer_size_finetuning_token

        # divide by 2 to be certain to fill the buffer in a single batch even if some tokens are padding tokens
        # wandb table initialiser
        self.wandb_sequence_expert_usage_table_initialized = False
        self.wandb_token_expert_usage_table_initialized = False

        self.correlation_tokens_scatter_plots = []
        self.correlation_sequences_scatter_plots = []

        # clean up some of the attributes: variables which should not be attributes are made attributes

    # def log_layer_level_metrics(self,gate_logits: tuple[torch.Tensor], attention_mask: torch.Tensor, cfg: DictConfig,step:int):

    #     #metrics
    #     self.mean_expert_usage_per_layer, self.max_expert_usage_per_layer = self.get_mean_and_max_expert_usage_per_layer(gate_logits, attention_mask, cfg)
    #     self.correlation_mean_expert_usage_across_layers = self.get_correlation_mean_expert_usage_across_layers(self.mean_expert_usage_per_layer)

    #     # Tables for per layer expert usage
    #     n_layers = len(self.mean_expert_usage_per_layer)
    #     if not self.wandb_expert_usage_table_initialized:
    #         columns = ["step"] + [f"layer_{i}" for i in range(n_layers)]
    #         self.mean_expert_usage_table = wandb.Table(columns=columns)
    #         self.max_expert_usage_table = wandb.Table(columns=columns)
    #         self.wandb_expert_usage_table_initialized = True

    #     # Add new rows for this step
    #     mean_row = [int(step)] + [float(self.mean_expert_usage_per_layer[i].item()) for i in range(n_layers)]
    #     max_row = [int(step)] + [float(self.max_expert_usage_per_layer[i].item()) for i in range(n_layers)]
    #     self.mean_expert_usage_table.add_data(*mean_row)
    #     self.max_expert_usage_table.add_data(*max_row)

    #     #
    # ation metric
    #     self.accelerator.log({
    #         "train/correlation_mean_expert_usage_across_layers": self.correlation_mean_expert_usage_across_layers.item()
    #     })

    def log_expert_usage_per_layer_plots(self, metrics_dict: defaultdict):
        """
        Logs the expert usage per layer DataFrame to wandb.
        """
        if hasattr(self, "max_expert_usage_per_layer_df"):
            max_columns = self.max_expert_usage_per_layer_df.columns
            xs = self.max_expert_usage_per_layer_df[max_columns[0]].tolist()
            ys = [
                self.max_expert_usage_per_layer_df[col].tolist()
                for col in max_columns[1:]
            ]
            metrics_dict["train/max_expert_usage_per_layer"] = wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=max_columns[1:],
                title="Max Expert Usage Per Layer",
                xname=max_columns[0],
            )
            # self.accelerator.log({
            #     "train/max_expert_usage_per_layer": wandb.plot.line_series(
            #         xs=xs,
            #         ys=ys,
            #         keys=max_columns[1:],
            #         title="Max Expert Usage Per Layer",
            #         xname=max_columns[0],
            #     )
            # })
        if hasattr(self, "mean_expert_usage_per_layer_df"):
            mean_columns = self.mean_expert_usage_per_layer_df.columns
            xs = self.mean_expert_usage_per_layer_df[mean_columns[0]].tolist()
            ys = [
                self.mean_expert_usage_per_layer_df[col].tolist()
                for col in mean_columns[1:]
            ]
            metrics_dict["train/mean_expert_usage_per_layer"] = wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=mean_columns[1:],
                title="Mean Expert Usage Per Layer",
                xname=mean_columns[0],
            )
            # self.accelerator.log({
            #     "train/mean_expert_usage_per_layer": wandb.plot.line_series(
            #         xs=xs,
            #         ys=ys,
            #         keys=mean_columns[1:],
            #         title="Mean Expert Usage Per Layer",
            #         xname=mean_columns[0],
            #     )
            # })

    def log_different_expert_usage_consecutive_layers(
        self,
        gate_logits: tuple[torch.Tensor],
        attention_mask: torch.Tensor,
        cfg: DictConfig,
        metrics_dict: defaultdict,
    ):
        # metrics
        mean_expert_usage_per_layer_per_token, _ = (
            self._get_mean_and_max_expert_usage_per_layer_per_token(
                gate_logits, attention_mask, cfg
            )
        )
        self.difference_mean_expert_usage_across_layers = (
            self.get_difference_mean_expert_usage_across_layers(
                mean_expert_usage_per_layer_per_token
            )
        )

        metrics_dict["train/difference_mean_expert_usage_across_layers"] = (
            self.difference_mean_expert_usage_across_layers.item()
        )

        # self.accelerator.log({
        #     "train/difference_mean_expert_usage_across_layers": self.difference_mean_expert_usage_across_layers.item()
        # })

    def append_expert_usage_across_layers(
        self,
        gate_logits: tuple[torch.Tensor],
        attention_mask: torch.Tensor,
        cfg: DictConfig,
        step: int,
    ):
        """
        Appends a new row to a DataFrame (self.expert_usage_proportions_df) containing expert usage proportions.
        Initializes the DataFrame if it does not exist.
        """
        # metrics
        self.mean_expert_usage_per_layer, self.max_expert_usage_per_layer = (
            self.get_mean_and_max_expert_usage_per_layer(
                gate_logits, attention_mask, cfg
            )
        )
        # Tables for per layer expert usage
        n_layers = len(self.mean_expert_usage_per_layer)

        columns = ["step"] + [f"layer_{i}" for i in range(n_layers)]
        mean_row = [step] + [
            self.mean_expert_usage_per_layer[i].item() for i in range(n_layers)
        ]
        max_row = [step] + [
            self.max_expert_usage_per_layer[i].item() for i in range(n_layers)
        ]
        if (
            not hasattr(self, "mean_expert_usage_per_layer_df")
            or self.mean_expert_usage_per_layer_df is None
        ):
            self.mean_expert_usage_per_layer_df = pd.DataFrame(columns=columns)
        if (
            not hasattr(self, "max_expert_usage_per_layer_df")
            or self.max_expert_usage_per_layer_df is None
        ):
            self.max_expert_usage_per_layer_df = pd.DataFrame(columns=columns)
        # Append the new row as a DataFrame and ignore index to keep appending
        self.mean_expert_usage_per_layer_df = pd.concat(
            [
                self.mean_expert_usage_per_layer_df,
                pd.DataFrame([mean_row], columns=columns),
            ],
            ignore_index=True,
        )
        self.max_expert_usage_per_layer_df = pd.concat(
            [
                self.max_expert_usage_per_layer_df,
                pd.DataFrame([max_row], columns=columns),
            ],
            ignore_index=True,
        )

    def log_expert_usage_proportions_plots(self, metrics_dict: defaultdict):
        """
        Logs the expert usage proportions DataFrame to wandb.
        """
        if hasattr(self, "expert_usage_proportions_df"):
            prop_columns = self.expert_usage_proportions_df.columns
            xs = self.expert_usage_proportions_df[prop_columns[0]].tolist()
            ys = [
                self.expert_usage_proportions_df[col].tolist()
                for col in prop_columns[1:]
            ]

            metrics_dict["train/expert_usage_proportions"] = wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=prop_columns[1:],
                title="Expert Usage Proportions",
                xname=prop_columns[0],
            )
            # self.accelerator.log({
            #     "train/expert_usage_proportions": wandb.plot.line_series(
            #         xs=xs,
            #         ys=ys,
            #         keys=prop_columns[1:],
            #         title="Expert Usage Proportions",
            #         xname=prop_columns[0]
            #     )
            # })

    def append_expert_level_metrics(
        self,
        gate_logits: tuple[torch.Tensor],
        attention_mask: torch.Tensor,
        cfg: DictConfig,
        step: int,
    ):
        """
        Appends a new row to a DataFrame (self.expert_usage_proportions_df) containing expert usage proportions.
        Initializes the DataFrame if it does not exist.
        """
        expert_usage_cum_prob = self.get_expert_usage_cum_prob(
            gate_logits, attention_mask, cfg
        )
        expert_usage_proportions = expert_usage_cum_prob / torch.sum(
            expert_usage_cum_prob
        )

        n_experts = expert_usage_proportions.shape[0]
        columns = ["step"] + [f"expert_{i}" for i in range(n_experts)]
        row = [step] + [expert_usage_proportions[i].item() for i in range(n_experts)]
        if (
            not hasattr(self, "expert_usage_proportions_df")
            or self.expert_usage_proportions_df is None
        ):
            self.expert_usage_proportions_df = pd.DataFrame(columns=columns)
        # Append the new row as a DataFrame and ignore index to keep appending
        self.expert_usage_proportions_df = pd.concat(
            [self.expert_usage_proportions_df, pd.DataFrame([row], columns=columns)],
            ignore_index=True,
        )

    # def log_expert_level_metrics(self,gate_logits: tuple[torch.Tensor], attention_mask: torch.Tensor, cfg: DictConfig,step:int):

    #     expert_usage_cum_prob = self.get_expert_usage_cum_prob(gate_logits, attention_mask, cfg)
    #     expert_usage_proportions = expert_usage_cum_prob / torch.sum(expert_usage_cum_prob)
    #     # Tables for per layer expert usage
    #     n_experts = len(expert_usage_cum_prob)
    #     if not self.wandb_expert_usage_proportions_table_initialized:
    #         columns = ["step"] + [f"expert_{i}" for i in range(n_experts)]
    #         self.expert_usage_proportions_table = wandb.Table(columns=columns)
    #         self.wandb_expert_usage_proportions_table_initialized = True

    #     # Add new rows for this step
    #     row = [step] + [expert_usage_proportions[i].item() for i in range(n_experts)]
    #     self.expert_usage_proportions_table.add_data(*row)

    #     self._append_expert_level_metrics(step, expert_usage_proportions)

    def accumulate_ce_and_expert_usage_token(
        self,
        logits: torch.Tensor,
        gate_logits: tuple[torch.Tensor],
        attention_mask: torch.Tensor,
        cfg: DictConfig,
        batch: dict,
        tokenizer,
    ):

        normalised_expert_usage_cost_per_token = (
            self.get_normalised_expert_usage_cost_per_masked_token(
                gate_logits, attention_mask, batch, cfg, tokenizer
            )
        )  # shape (num_valid_tokens_in_batch,)
        cross_entropy_loss_per_token = self.get_cross_entropy_per_masked_token(
            logits, cfg, batch, tokenizer
        )  # shape (num_valid_tokens_in_batch,)

        # Accumulate in buffers
        self.expert_usage_token_buffer.extend(
            normalised_expert_usage_cost_per_token.detach().cpu().tolist()
        )
        self.ce_loss_token_buffer.extend(
            cross_entropy_loss_per_token.detach().cpu().tolist()
        )

    def log_correlation_between_ce_and_expert_usage_token(
        self, metrics_dict: defaultdict, step, prefix="train"
    ):

        # When buffer is full, compute correlation and log, then reset
        if (
            len(self.expert_usage_token_buffer) >= self.buffer_size_token
            and len(self.ce_loss_token_buffer) >= self.buffer_size_token
        ):
            # Truncate to buffer_size in case of overflow
            # expert_usage_arr = torch.tensor(self.expert_usage_token_buffer[:self.buffer_size_token])
            # cross_entropy_loss_arr = torch.tensor(self.ce_loss_token_buffer[:self.buffer_size_token])

            expert_usage_list = self.expert_usage_token_buffer[: self.buffer_size_token]
            cross_entropy_loss_list = self.ce_loss_token_buffer[
                : self.buffer_size_token
            ]

            self._visualise_correlation(
                expert_usage_list,
                cross_entropy_loss_list,
                metrics_dict,
                x_axis_name="Cross Entropy Loss per Token",
                y_axis_name="Expert Usage per Token",
                title="Expert Usage vs Cross Entropy Loss",
                step=step,
                prefix=prefix,
            )

            expert_usage_arr = torch.tensor(expert_usage_list)
            cross_entropy_loss_arr = torch.tensor(cross_entropy_loss_list)

            # Gave up on this: not as nice to use wandb.plot.scatter plots. Better to stack images

            #     data = list(zip(centered_expert_usage_list, centered_cross_entropy_loss_list))

            #     table = wandb.Table(data=data, columns=["centered_expert_usage", "centered_cross_entropy_loss"])

            #     scatter_plot = wandb.plot.scatter(table=table,
            #         x="centered_cross_entropy_loss",
            #         y="centered_expert_usage",
            #         title="Expert Usage vs Cross Entropy Loss",
            # )

            # Compute Pearson correlation

            if expert_usage_arr.std() > 0 and cross_entropy_loss_arr.std() > 0:

                correlation = torch.corrcoef(
                    torch.stack([expert_usage_arr, cross_entropy_loss_arr])
                )[0, 1].item()

            else:
                correlation = 0.0
            # Log correlation
            metrics_dict[f"{prefix}/expert_usage_cross_entropy_per_token_corr"] = (
                correlation
            )
            # self.accelerator.log({f"{prefix}/expert_usage_cross_entropy_per_token_corr": correlation})
            # Reset buffers
            self.expert_usage_token_buffer = []
            self.ce_loss_token_buffer = []

    def _visualise_correlation(
        self,
        expert_usage_list,
        task_loss_list,
        metrics_dict,
        x_axis_name="Cross Entropy Loss per Token",
        y_axis_name="Expert Usage per Token",
        title="Expert Usage vs Task Loss",
        step=None,
        prefix="train",
    ):
        mean_expert_usage = sum(expert_usage_list) / len(expert_usage_list)
        mean_task_loss = sum(task_loss_list) / len(task_loss_list)
        centered_expert_usage_list = [x - mean_expert_usage for x in expert_usage_list]
        centered_task_loss_list = [x - mean_task_loss for x in task_loss_list]

        plt.scatter(centered_task_loss_list, centered_expert_usage_list, alpha=0.05)
        plt.xlabel("Centered " + x_axis_name)
        plt.ylabel("Centered " + y_axis_name)
        plt.title(title + "_" + str(step))
        scatter_plot_filename = title + "_" + str(step) + ".png"
        plt.savefig(scatter_plot_filename)
        plt.close()

        plt.scatter(task_loss_list, centered_expert_usage_list, alpha=0.05)
        x_axis_name = "Centered " + x_axis_name
        plt.xlabel("Original " + x_axis_name)
        plt.ylabel("Centered " + y_axis_name)
        # here I want all plots to have the same x and y limits so that they can be compared
        plt.xlim(-1, 20)
        plt.ylim(-0.2, 0.2)
        plt.title(title + " with Fixed Bounds " + str(step))
        scatter_plot_filename_fixed = title + "_fixed_bounds" + "_" + str(step) + ".png"
        plt.savefig(scatter_plot_filename_fixed)
        plt.close()

        # plt.scatter(centered_cross_entropy_loss_list, centered_expert_usage_list, alpha = 0.2)
        # plt.xlabel("Centered Cross Entropy Loss per Token")
        # plt.ylabel("Centered Expert Usage per Token")
        # #here I want all plots to have the same x and y limits so that they can be compared
        # plt.xlim(-12, 12)
        # plt.ylim(-0.2, 0.2)
        # plt.title("Scatter Plot: Expert Usage vs Cross Entropy Loss with Fixed Bounds "+ str(metrics_dict["train/steps"]))
        # scatter_plot_filename_fixed = "expert_usage_vs_ce_loss_fixed_bounds" + "_" + str(metrics_dict["train/steps"])+".png"
        # plt.savefig(scatter_plot_filename_fixed)
        # plt.close()
        # self.correlation_tokens_scatter_plots.append(wandb.Image(scatter_plot_filename, caption="Expert Usage vs Cross Entropy Loss Scatter Plot"))
        metrics_dict[f"{prefix}/{title}"] = wandb.Image(scatter_plot_filename)
        metrics_dict[f"{prefix}/{title}_fixed_bounds"] = wandb.Image(
            scatter_plot_filename_fixed
        )

    def accumulate_ce_and_expert_usage_seq(
        self,
        logits: torch.Tensor,
        gate_logits: tuple[torch.Tensor],
        attention_mask: torch.Tensor,
        cfg: DictConfig,
        batch: dict,
    ):

        normalised_expert_usage_cost_per_seq = (
            self.get_normalised_expert_usage_cost_per_sequence(
                gate_logits, attention_mask, cfg
            )
        )  # shape (batch_size,)
        cross_entropy_loss_per_seq = self.get_cross_entropy_per_sequence(
            logits, cfg, batch
        )  # batch size

        # Accumulate in buffers
        self.expert_usage_seq_buffer.extend(
            normalised_expert_usage_cost_per_seq.detach().cpu().tolist()
        )
        self.ce_loss_seq_buffer.extend(
            cross_entropy_loss_per_seq.detach().cpu().tolist()
        )

    def log_correlation_between_ce_and_expert_usage_seq(
        self, metrics_dict: defaultdict, prefix="train"
    ):

        # When buffer is full, compute correlation and log, then reset
        if (
            len(self.expert_usage_seq_buffer) >= self.buffer_size_seq
            and len(self.ce_loss_seq_buffer) >= self.buffer_size_seq
        ):
            # Truncate to buffer_size in case of overflow
            expert_usage_arr = torch.tensor(
                self.expert_usage_seq_buffer[: self.buffer_size_seq]
            )
            cross_entropy_loss_arr = torch.tensor(
                self.ce_loss_seq_buffer[: self.buffer_size_seq]
            )
            # Compute Pearson correlation
            # metrics_dict[f"{prefix}/expert_usage_per_seq"] = expert_usage_arr
            # metrics_dict[f"{prefix}/cross_entropy_per_seq"] = cross_entropy_loss_arr
            if expert_usage_arr.std() > 0 and cross_entropy_loss_arr.std() > 0:
                correlation = torch.corrcoef(
                    torch.stack([expert_usage_arr, cross_entropy_loss_arr])
                )[0, 1].item()
            else:
                correlation = 0.0
            # Log correlation
            metrics_dict[f"{prefix}/expert_usage_cross_entropy_per_seq_corr"] = (
                correlation
            )
            # self.accelerator.log({f"{prefix}/expert_usage_cross_entropy_per_seq_corr": correlation})
            # Reset buffers
            self.expert_usage_seq_buffer = []
            self.ce_loss_seq_buffer = []

    # for finetuning where the task loss is not necessarily cross entropy and we have just a loss at sequence level
    def accumulate_task_loss_and_expert_usage_seq(
        self,
        logits: torch.Tensor,
        gate_logits: tuple[torch.Tensor],
        attention_mask: torch.Tensor,
        cfg: DictConfig,
        batch: dict,
        is_regression,
        num_labels: int,
    ):

        normalised_expert_usage_cost_per_seq = (
            self.get_normalised_expert_usage_cost_per_sequence(
                gate_logits, attention_mask, cfg
            )
        )  # shape (batch_size,)
        task_loss_per_seq = self.get_task_loss_per_sequence(
            logits, batch, is_regression, num_labels
        )  # batch size
        # print("normalised_expert_usage_cost_per_seq:", normalised_expert_usage_cost_per_seq)
        # print("normalised_expert_usage_cost_per_seq shape:", normalised_expert_usage_cost_per_seq.shape)
        # print("task_loss_per_seq:", task_loss_per_seq)
        # print("task_loss_per_seq shape:", task_loss_per_seq.shape)

        # Accumulate in buffers
        self.expert_usage_seq_buffer_finetuning.extend(
            normalised_expert_usage_cost_per_seq.detach().cpu().tolist()
        )
        self.task_loss_seq_buffer_finetuning.extend(
            task_loss_per_seq.detach().cpu().tolist()
        )
        # print(len(self.expert_usage_seq_buffer_finetuning))
        # print(len(self.task_loss_seq_buffer_finetuning))

    def accumulate_task_loss_and_expert_usage_token(
        self,
        logits: torch.Tensor,
        gate_logits: tuple[torch.Tensor],
        attention_mask: torch.Tensor,
        cfg: DictConfig,
        batch: dict,
        is_regression,
        num_labels: int,
    ):

        normalised_expert_usage_cost_per_token = (
            self.get_normalised_expert_usage_cost_per_token(
                gate_logits, attention_mask, batch, cfg, self.tokenizer
            )
        )  # shape (batch_size,seq_length)
        task_loss_per_token = self.get_task_loss_per_token(
            logits, batch, is_regression, num_labels, attention_mask
        )  # batch size*seq_length
        task_loss_per_token = task_loss_per_token.reshape(
            -1
        )  # Flatten to (num_valid_tokens_in_batch,)

        # filter out the padding tokens
        attention_mask_flat = attention_mask.reshape(
            -1
        )  # Flatten to (batch_size * seq_length,)
        task_loss_per_token = task_loss_per_token[attention_mask_flat == 0]
        normalised_expert_usage_cost_per_token = normalised_expert_usage_cost_per_token[
            attention_mask_flat == 0
        ]

        # print("normalised_expert_usage_cost_per_token:", normalised_expert_usage_cost_per_token)
        # print("normalised_expert_usage_cost_per_token shape:", normalised_expert_usage_cost_per_token.shape)
        # print("task_loss_per_token:", task_loss_per_token)
        # print("task_loss_per_token shape:", task_loss_per_token.shape)

        # Accumulate in buffers
        self.expert_usage_token_buffer_finetuning.extend(
            normalised_expert_usage_cost_per_token.detach().cpu().tolist()
        )
        self.task_loss_token_buffer_finetuning.extend(
            task_loss_per_token.detach().cpu().tolist()
        )

        # print(len(self.expert_usage_token_buffer_finetuning))
        # print(len(self.task_loss_token_buffer_finetuning))
        # print("buffer size finetuning token:", self.buffer_size_finetuning_token)

    def log_correlation_between_task_loss_and_expert_usage_seq(
        self, metrics_dict: defaultdict, step, prefix="train"
    ):

        # When buffer is full, compute correlation and log, then reset
        if (
            len(self.expert_usage_seq_buffer_finetuning)
            >= self.buffer_size_finetuning_seq
            and len(self.task_loss_seq_buffer_finetuning)
            >= self.buffer_size_finetuning_seq
        ):
            # Truncate to buffer_size in case of overflow

            expert_usage_list = self.expert_usage_seq_buffer_finetuning[
                : self.buffer_size_finetuning_seq
            ]
            task_loss_list = self.task_loss_seq_buffer_finetuning[
                : self.buffer_size_finetuning_seq
            ]

            self._visualise_correlation(
                expert_usage_list,
                task_loss_list,
                metrics_dict=metrics_dict,
                x_axis_name="Expert Usage per seq",
                y_axis_name="Task Loss per seq",
                title="Correlation between Expert Usage and Task Loss per seq",
                step=step,
                prefix=prefix,
            )

            expert_usage_arr = torch.tensor(
                self.expert_usage_seq_buffer_finetuning[
                    : self.buffer_size_finetuning_seq
                ]
            )
            task_loss_arr = torch.tensor(
                self.task_loss_seq_buffer_finetuning[: self.buffer_size_finetuning_seq]
            )
            # Compute Pearson correlations
            # metrics_dict[f"{prefix}/expert_usage_per_seq"] = expert_usage_arr
            # metrics_dict[f"{prefix}/cross_entropy_per_seq"] = cross_entropy_loss_arr
            if expert_usage_arr.std() > 0 and task_loss_arr.std() > 0:
                correlation = torch.corrcoef(
                    torch.stack([expert_usage_arr, task_loss_arr])
                )[0, 1].item()
            else:
                correlation = 0.0

            # Log correlation
            metrics_dict[f"{prefix}/expert_usage_task_loss_per_seq_corr"] = correlation
            # self.accelerator.log({f"{prefix}/expert_usage_task_loss_per_seq_corr": correlation})
            # Reset buffers
            self.expert_usage_seq_buffer_finetuning = []
            self.task_loss_seq_buffer_finetuning = []

    def log_correlation_between_task_loss_and_expert_usage_token(
        self, metrics_dict: defaultdict, step, prefix="train"
    ):

        # When buffer is full, compute correlation and log, then reset
        if (
            len(self.expert_usage_token_buffer_finetuning)
            >= self.buffer_size_finetuning_token
            and len(self.task_loss_token_buffer_finetuning)
            >= self.buffer_size_finetuning_token
        ):
            # Truncate to buffer_size in case of overflow

            expert_usage_list = self.expert_usage_token_buffer_finetuning[
                : self.buffer_size_finetuning_token
            ]
            task_loss_list = self.task_loss_token_buffer_finetuning[
                : self.buffer_size_finetuning_token
            ]

            self._visualise_correlation(
                expert_usage_list,
                task_loss_list,
                metrics_dict,
                x_axis_name="Expert Usage per Token",
                y_axis_name="Task Loss per Token",
                title="Correlation between Expert Usage and Task Loss per Token",
                step=step,
                prefix=prefix,
            )

            expert_usage_arr = torch.tensor(
                self.expert_usage_token_buffer_finetuning[
                    : self.buffer_size_finetuning_token
                ]
            )
            task_loss_arr = torch.tensor(
                self.task_loss_token_buffer_finetuning[
                    : self.buffer_size_finetuning_token
                ]
            )
            # Compute Pearson correlations
            # metrics_dict[f"{prefix}/expert_usage_per_seq"] = expert_usage_arr
            # metrics_dict[f"{prefix}/cross_entropy_per_seq"] = cross_entropy_loss_arr
            if expert_usage_arr.std() > 0 and task_loss_arr.std() > 0:
                correlation = torch.corrcoef(
                    torch.stack([expert_usage_arr, task_loss_arr])
                )[0, 1].item()
            else:
                correlation = 0.0

            # Log correlation
            metrics_dict[f"{prefix}/expert_usage_task_loss_per_token_corr"] = (
                correlation
            )
            # self.accelerator.log({f"{prefix}/expert_usage_task_loss_per_seq_corr": correlation})
            # Reset buffers
            self.expert_usage_token_buffer_finetuning = []
            self.task_loss_token_buffer_finetuning = []

    def log_entropy_metrics(
        self,
        gate_logits: tuple[torch.Tensor],
        attention_mask: torch.Tensor,
        cfg: DictConfig,
        metrics_dict: defaultdict,
    ):

        self.entropy = self.get_entropy(gate_logits, attention_mask, cfg)
        self.entropy_per_sequence = self.get_entropy_per_sequence(
            gate_logits, attention_mask, cfg
        )
        metrics_dict["train/entropy_per_sequence"] = self.entropy_per_sequence.item()
        metrics_dict["train/entropy"] = self.entropy.item()

        # self.accelerator.log({"train/entropy_per_sequence": self.entropy_per_sequence.item()})
        # self.accelerator.log({"train/entropy": self.entropy.item()})

    def log_std_rel_metrics(
        self,
        logits: torch.Tensor,
        gate_logits: tuple[torch.Tensor],
        attention_mask: torch.Tensor,
        cfg: DictConfig,
        metrics_dict: defaultdict,
    ):
        # metrics

        self.intra_sequence_normalised_expert_cost_rel_std = (
            self.get_intra_sequence_normalised_expert_cost_rel_std(
                gate_logits, attention_mask, cfg
            )
        )
        (
            self.inter_token_normalised_expert_cost_rel_std,
            self.inter_token_normalised_expert_cost_std,
            self.inter_token_normalised_expert_cost_mean,
        ) = self.get_inter_token_normalised_expert_cost_rel_std(
            gate_logits, attention_mask, cfg
        )
        self.inter_sequence_normalised_expert_cost_rel_std = (
            self.get_inter_sequence_normalised_expert_cost_rel_std(
                gate_logits, attention_mask, cfg
            )
        )

        # log metrics
        metrics_dict["train/intra_sequence_normalised_expert_cost_rel_std"] = (
            self.intra_sequence_normalised_expert_cost_rel_std.item()
        )
        metrics_dict["train/inter_token_normalised_expert_cost_rel_std"] = (
            self.inter_token_normalised_expert_cost_rel_std.item()
        )
        metrics_dict["train/inter_token_normalised_expert_cost_std"] = (
            self.inter_token_normalised_expert_cost_std.item()
        )
        metrics_dict["train/inter_token_normalised_expert_cost_mean"] = (
            self.inter_token_normalised_expert_cost_mean.item()
        )
        metrics_dict["train/inter_sequence_normalised_expert_cost_rel_std"] = (
            self.inter_sequence_normalised_expert_cost_rel_std.item()
        )

        # self.accelerator.log({
        #     "train/intra_sequence_normalised_expert_cost_rel_std": self.intra_sequence_normalised_expert_cost_rel_std.item(),
        #     "train/inter_token_normalised_expert_cost_rel_std": self.inter_token_normalised_expert_cost_rel_std.item(),
        #     "train/inter_sequence_normalised_expert_cost_rel_std": self.inter_sequence_normalised_expert_cost_rel_std.item(),
        # })

    def build_token_expert_usage_dic(
        self,
        batch_input_id: torch.Tensor,
        batch_gate_logits: tuple[torch.Tensor],
        batch_attention_mask: torch.Tensor,
        cfg: DictConfig,
    ):
        self.token_expert_usage_dict = self._append_expert_usage_per_token(
            self.token_expert_usage_dict,
            batch_input_id,
            batch_gate_logits,
            batch_attention_mask,
            cfg,
        )

    def build_sequence_expert_usage_dic(
        self,
        batch_input_id: torch.Tensor,
        batch_gate_logits: tuple[torch.Tensor],
        batch_attention_mask: torch.Tensor,
        cfg: DictConfig,
    ):
        self.sequence_expert_usage_dict = self._append_expert_usage_per_sequence(
            self.sequence_expert_usage_dict,
            batch_input_id,
            batch_gate_logits,
            batch_attention_mask,
            cfg,
        )

    def log_token_expert_usage_df(
        self, token_expert_usage_dict: defaultdict, metrics_dict: defaultdict
    ):
        self.df_of_average_token_usage = self.get_df_of_average_token_usage(
            token_expert_usage_dict, self.tokenizer
        )
        # log table
        if not self.wandb_token_expert_usage_table_initialized:
            columns = [
                "token_id",
                "written_token",
                "avg_expert_usage",
                "rel_std_expert_usage",
            ]
            self.token_expert_usage_table = wandb.Table(columns=columns)
            self.wandb_token_expert_usage_table_initialized = True

        # Add new rows for this step
        n_rows = self.df_of_average_token_usage.shape[0]
        for i, (_, row) in enumerate(self.df_of_average_token_usage.iterrows()):
            if i < 100 or i > n_rows - 100:  # log only first 100 and last 100 rows
                self.token_expert_usage_table.add_data(
                    row["token_id"],
                    row["written_token"],
                    row["avg_expert_usage"],
                    row["rel_std_expert_usage"],
                )
        metrics_dict["token_expert_usage"] = self.token_expert_usage_table
        # self.accelerator.log({
        #     f"token_expert_usage:{suffix}": self.token_expert_usage_table
        # })

    def log_sequence_expert_usage_df(
        self, sequence_expert_usage_dict: defaultdict, metrics_dict: defaultdict
    ):
        self.df_of_average_sequence_usage = self.get_df_of_average_sequence_usage(
            sequence_expert_usage_dict, self.tokenizer
        )
        # log table
        if not self.wandb_sequence_expert_usage_table_initialized:
            columns = ["seq_id", "written_seq", "avg_expert_usage"]
            self.sequence_expert_usage_table = wandb.Table(columns=columns)
            self.wandb_sequence_expert_usage_table_initialized = True

        # Add new rows for this step
        n_rows = self.df_of_average_sequence_usage.shape[0]
        for i, (_, row) in enumerate(self.df_of_average_sequence_usage.iterrows()):
            if i < 100 or i > n_rows - 100:  # log only first 100 and last 100 rows
                self.sequence_expert_usage_table.add_data(
                    row["seq_id"], row["written_seq"], row["avg_expert_usage"]
                )

        metrics_dict["sequence_expert_usage"] = self.sequence_expert_usage_table
        # self.accelerator.log({
        #     f"sequence_expert_usage:{suffix}": self.sequence_expert_usage_table
        # })

    def _visualise_individual_token_path(
        self,
        gate_logits,
        token_index,
        token_str,
        visualisations_list,
        sequence_str=None,
    ):
        # Stack logits if tuple
        if isinstance(gate_logits, tuple):
            compute_device = gate_logits[0].device
            target_stacked_gate_logits = torch.stack(
                [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
            )  # n_layers, batch_size*seq_thth, n_experts
        else:
            target_stacked_gate_logits = gate_logits
        target_routing_weights = torch.nn.functional.softmax(
            target_stacked_gate_logits, dim=-1
        )  # n_layers, batch_size*seq_length, n_experts

        # Select the routing weights for the given token index
        # Shape: (n_layers, n_experts)
        target_weights_token = (
            target_routing_weights[:, token_index, :].detach().cpu().numpy()
        )

        n_layers, n_experts = target_weights_token.shape

        # Compute top-2 experts for each layer for target and prediction
        target_top2 = np.argsort(target_weights_token, axis=1)[
            :, -2:
        ]  # shape (n_layers, 2)

        # Plot side-by-side heatmaps
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(target_weights_token, aspect="auto", cmap="viridis")
        ax.set_title("Target Routing Weights")
        ax.set_xlabel("Expert")
        ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add black circles for target top-2 experts
        for layer in range(n_layers):
            # Black circles for target
            for expert in target_top2[layer]:
                ax.scatter(
                    expert,
                    layer,
                    s=80,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=2,
                    marker="o",
                    zorder=3,
                )

        if sequence_str is not None:
            # fig.suptitle(f"sequence:{sequence_str}, token:'{token_str}'")
            fig.suptitle(f"token:{token_str}")
        else:
            fig.suptitle(f"token:{token_str}")
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

        black_circle = mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            markersize=8,
            label="Target top-2 expert",
        )

        handles = [black_circle]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=2,
            fontsize="small",
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )

        # Save to buffer and append to visualisations_list
        figure_filename = f"token_path_for:{token_index}.png"
        plt.savefig(figure_filename)
        plt.close(fig)

        self.visualisations_list.append(wandb.Image(figure_filename))

    def visualise_token_routing_paths(
        self, batch, gate_logits, metrics_dict, num_visualisations=50
    ):
        """Visualise the routing patterns for a few random tokens in a few random samples from a batch
        There is a risk to see a lot of padding tokens if the dataset has short sequences.

        """

        tokenizer = self.tokenizer

        self.visualisations_list = []
        # For 10 different random tokens in batch*seq_length, pick a token index,
        # extract the token _id from this,
        # decode it to the original token through the tokenizer,
        # and visualise the routing path.

        input_ids = batch["input_ids"]  # shape (batch_size, seq_length)
        batch_size, seq_length = input_ids.shape

        # seq_indices = np.random.choice(batch_size, num_visualisations)
        # token_indices = np.random.choice(seq_length, num_visualisations)
        visualisation_index = 0
        while len(self.visualisations_list) < num_visualisations:
            seq_index = np.random.choice(batch_size)
            token_index = np.random.choice(seq_length)
            token_id = input_ids[seq_index, token_index].item()
            if token_id == tokenizer.pad_token_id:
                # Skip padding tokens but not strictly, keep a few of them
                continue

            token_str = tokenizer.decode([token_id])
            # except Exception:
            token_str = tokenizer.decode([token_id])
            sequence_str = tokenizer.decode(input_ids[seq_index].tolist())
            self._visualise_individual_token_path(
                gate_logits,
                seq_index * seq_length + token_index,
                token_str,
                self.visualisations_list,
                sequence_str=sequence_str,
            )

        metrics_dict["test/visualisations"] = self.visualisations_list


class AnalysisTraining(AnalysisLogger):
    def __init__(
        self,
        cfg,
        accelerator,
    ):
        super().__init__(cfg, accelerator)
        self.sub_step = 0  # in order to sample the large expert usage plots at a lower frequency than the other metrics and not overload wandb

    def __call__(self, batch, model_output, step, metrics_dict):

        self.step = step
        self.attention_mask = batch.get("attention_mask", None)
        self.batch = batch
        self.model_output = model_output
        self.gate_logits = model_output["router_logits"]
        self.logits = model_output["logits"]

        # at each batch
        # self.build_token_expert_usage_dic(self.batch["input_ids"], self.gate_logits, self.attention_mask, self.cfg)
        # self.build_sequence_expert_usage_dic(self.batch["input_ids"], self.gate_logits, self.attention_mask, self.cfg)
        # self.accumulate_ce_and_expert_usage_seq(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch)
        # self.accumulate_ce_and_expert_usage_token(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch,self.tokenizer)
        # self.log_correlation_between_ce_and_expert_usage_seq(metrics_dict)
        # self.log_correlation_between_ce_and_expert_usage_token(metrics_dict)

        # at each log step
        if step % self.cfg.wandb.log_interval == 0:
            self.sub_step += 1
            if self.sub_step % 10 == 0:
                self.sub_step = 0
                self.append_expert_usage_across_layers(
                    self.gate_logits, self.attention_mask, self.cfg, self.step
                )
                self.append_expert_level_metrics(
                    self.gate_logits, self.attention_mask, self.cfg, self.step
                )
            self.log_different_expert_usage_consecutive_layers(
                self.gate_logits, self.attention_mask, self.cfg, metrics_dict
            )
            self.log_entropy_metrics(
                self.gate_logits, self.attention_mask, self.cfg, metrics_dict
            )
            self.log_std_rel_metrics(
                self.logits,
                self.gate_logits,
                self.attention_mask,
                self.cfg,
                metrics_dict,
            )

            self.accumulate_ce_and_expert_usage_seq(
                self.logits, self.gate_logits, self.attention_mask, self.cfg, self.batch
            )
            self.accumulate_ce_and_expert_usage_token(
                self.logits,
                self.gate_logits,
                self.attention_mask,
                self.cfg,
                self.batch,
                self.tokenizer,
            )
            self.log_correlation_between_ce_and_expert_usage_seq(metrics_dict)
            self.log_correlation_between_ce_and_expert_usage_token(
                metrics_dict, step=metrics_dict["train/steps"]
            )

            # self.build_token_expert_usage_dic(self.batch["input_ids"], self.gate_logits, self.attention_mask, self.cfg)
            # self.build_sequence_expert_usage_dic(self.batch["input_ids"], self.gate_logits, self.attention_mask, self.cfg)
            # self.log_correlation_between_ce_and_expert_usage_seq(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch)
            # self.log_correlation_between_ce_and_expert_usage_seq(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch)
            # self.log_correlation_between_ce_and_expert_usage_token(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch,self.tokenizer)
            self.log_expert_usage_per_layer_plots(metrics_dict)
            self.log_expert_usage_proportions_plots(metrics_dict)

        # only at the end of training
        if step == self.cfg.trainer.max_steps - 1:
            print("logging final analysis results to wandb")

            # self.log_sequence_expert_usage_df(self.sequence_expert_usage_dict, metrics_dict)
            # self.log_token_expert_usage_df(self.token_expert_usage_dict, metrics_dict)
            # self.log_expert_usage_per_layer_plots(metrics_dict)
            # self.log_expert_usage_proportions_plots(metrics_dict)


class AnalysisTrainedModel(AnalysisLogger):
    def __init__(self, cfg, accelerator, max_steps, buffer_size_seq, buffer_size_token):
        super().__init__(cfg, accelerator, buffer_size_seq, buffer_size_token)
        self.max_steps = max_steps

    def __call__(self, batch, model_output, step, metrics_dict):

        self.step = step
        self.attention_mask = batch.get("attention_mask", None)
        self.batch = batch
        self.model_output = model_output
        self.gate_logits = model_output["router_logits"]
        self.logits = model_output["logits"]

        # at each batch
        self.build_token_expert_usage_dic(
            self.batch["input_ids"], self.gate_logits, self.attention_mask, self.cfg
        )
        self.build_sequence_expert_usage_dic(
            self.batch["input_ids"], self.gate_logits, self.attention_mask, self.cfg
        )
        self.accumulate_ce_and_expert_usage_seq(
            self.logits, self.gate_logits, self.attention_mask, self.cfg, self.batch
        )
        self.accumulate_ce_and_expert_usage_token(
            self.logits,
            self.gate_logits,
            self.attention_mask,
            self.cfg,
            self.batch,
            self.tokenizer,
        )

        # only at the end of training
        if step == self.max_steps - 1:
            print("logging final analysis results to wandb")
            self.log_sequence_expert_usage_df(
                self.sequence_expert_usage_dict, metrics_dict
            )
            self.log_token_expert_usage_df(self.token_expert_usage_dict, metrics_dict)
            self.log_correlation_between_ce_and_expert_usage_seq(
                metrics_dict, prefix="test"
            )
            self.log_correlation_between_ce_and_expert_usage_token(
                metrics_dict, step=metrics_dict["train/steps"], prefix="test"
            )

            self.accelerator.log(metrics_dict)


class AnalysisFinetuning(AnalysisLogger):

    def __init__(self, cfg, accelerator):
        super().__init__(cfg, accelerator)
        self.training_sub_step = 0  # in order to sample the large expert usage plots at a lower frequency than the other metrics and not overload wandb

    def __call__(
        self,
        batch,
        model_output,
        eval_step,
        training_step,
        metrics_dict,
        is_regression,
        num_labels: int,
        eval_dataloader,
        training: bool,
    ):
        # you should input the training_metrics_dict as metrics_dict when training=True and the eval_metrics_dict when training=False
        self.eval_step = eval_step
        self.training_step = training_step
        self.attention_mask = batch.get("attention_mask", None)
        self.batch = batch
        self.model_output = model_output
        self.gate_logits = model_output["router_logits"]
        self.logits = model_output["logits"]
        self.training = training

        # update the finetuning buffer size depending on the size of the  dataloader such that we have one correlation computation per evaluation epoch
        if not self.training:
            self.buffer_size_finetuning_seq = len(eval_dataloader.dataset)
            self.buffer_size_finetuning_token = (
                len(eval_dataloader.dataset) * 10
            )  # assuming average seq length >10 tokens. We don't need it to be exact here

        # at each batch
        # self.build_token_expert_usage_dic(self.batch["input_ids"], self.gate_logits, self.attention_mask, self.cfg)
        # self.build_sequence_expert_usage_dic(self.batch["input_ids"], self.gate_logits, self.attention_mask, self.cfg)
        # self.accumulate_ce_and_expert_usage_seq(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch)
        # self.accumulate_ce_and_expert_usage_token(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch,self.tokenizer)
        # self.log_correlation_between_ce_and_expert_usage_seq(metrics_dict)
        # self.log_correlation_between_ce_and_expert_usage_token(metrics_dict)

        if self.training:
            training_metrics_dict = metrics_dict
            if self.training_step % self.cfg.wandb.log_interval == 0:
                # only append the  most expensive metrics every n log steps since wandb can get overloaded
                self.training_sub_step += 1
                if self.training_sub_step % 100 == 0:
                    self.append_expert_usage_across_layers(
                        self.gate_logits,
                        self.attention_mask,
                        self.cfg,
                        self.training_step,
                    )
                    self.append_expert_level_metrics(
                        self.gate_logits,
                        self.attention_mask,
                        self.cfg,
                        self.training_step,
                    )
                self.log_different_expert_usage_consecutive_layers(
                    self.gate_logits,
                    self.attention_mask,
                    self.cfg,
                    training_metrics_dict,
                )

                self.log_entropy_metrics(
                    self.gate_logits,
                    self.attention_mask,
                    self.cfg,
                    training_metrics_dict,
                )
                self.log_std_rel_metrics(
                    self.logits,
                    self.gate_logits,
                    self.attention_mask,
                    self.cfg,
                    training_metrics_dict,
                )

                self.log_expert_usage_per_layer_plots(training_metrics_dict)
                self.log_expert_usage_proportions_plots(training_metrics_dict)

                # log the the training metrics dict at the associated training step (completed step)
                self.accelerator.log(training_metrics_dict, step=self.training_step)
                # attention à avoir le bon step qui va avec
                # attention à bien donner en entrée les deux metrics dicts

        else:  # in eval mode plot visualisations and correlation metric
            eval_metrics_dict = metrics_dict
            if self.eval_step == 1:
                pass
                # self.visualise_token_routing_paths(self.batch, self.gate_logits, eval_metrics_dict)

            # We compute most metrics once per evaluation step
            # if self.eval_step == 1:
            #     # self.append_expert_usage_across_layers(self.gate_logits, self.attention_mask, self.cfg,self.training_step)
            #     # self.append_expert_level_metrics(self.gate_logits, self.attention_mask, self.cfg,self.training_step)
            #     # self.log_different_expert_usage_consecutive_layers(self.gate_logits, self.attention_mask, self.cfg, metrics_dict)

            #     # self.log_entropy_metrics(self.gate_logits, self.attention_mask, self.cfg, metrics_dict)
            #     # self.log_std_rel_metrics(self.logits,self.gate_logits, self.attention_mask, self.cfg, metrics_dict)

            #     # self.log_expert_usage_per_layer_plots(metrics_dict)
            #     # self.log_expert_usage_proportions_plots(metrics_dict)

            #     # self.accumulate_ce_and_expert_usage_seq(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch)
            #     # self.accumulate_ce_and_expert_usage_token(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch,self.tokenizer)
            #     # self.log_correlation_between_ce_and_expert_usage_seq(metrics_dict)
            #     # self.log_correlation_between_ce_and_expert_usage_token(metrics_dict)

            # accumulate the task loss and expert usage for correlation over the whole eval set
            self.accumulate_task_loss_and_expert_usage_seq(
                self.logits,
                self.gate_logits,
                self.attention_mask,
                self.cfg,
                self.batch,
                is_regression,
                num_labels,
            )
            self.accumulate_task_loss_and_expert_usage_token(
                self.logits,
                self.gate_logits,
                self.attention_mask,
                self.cfg,
                self.batch,
                is_regression,
                num_labels,
            )

            # self.accumulate_ce_and_expert_usage_token(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch,self.tokenizer)

            if self.eval_step == len(eval_dataloader) - 1:
                self.log_correlation_between_task_loss_and_expert_usage_seq(
                    eval_metrics_dict, step=self.training_step
                )
                self.log_correlation_between_task_loss_and_expert_usage_token(
                    eval_metrics_dict, step=self.training_step
                )

                # self.log_correlation_between_ce_and_expert_usage_token(metrics_dict)

                # self.build_token_expert_usage_dic(self.batch["input_ids"], self.gate_logits, self.attention_mask, self.cfg)
                # self.build_sequence_expert_usage_dic(self.batch["input_ids"], self.gate_logits, self.attention_mask, self.cfg)
                # self.log_correlation_between_ce_and_expert_usage_seq(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch)
                # self.log_correlation_between_ce_and_expert_usage_seq(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch)
                # self.log_correlation_between_ce_and_expert_usage_token(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch,self.tokenizer)

            # #only at the end of training
            # if step == self.cfg.trainer.max_steps - 1:
            #     print("logging final analysis results to wandb")

            #     # self.log_sequence_expert_usage_df(self.sequence_expert_usage_dict, metrics_dict)
            #     # self.log_token_expert_usage_df(self.token_expert_usage_dict, metrics_dict)
            #     # self.log_expert_usage_per_layer_plots(metrics_dict)
            #     # self.log_expert_usage_proportions_plots(metrics_dict)


class AnalysisTestTrainedModel(AnalysisLogger):
    def __init__(self, cfg, accelerator, max_steps):
        # def __init__(self,cfg,accelerator,max_steps,buffer_size_seq,buffer_size_token):
        # super().__init__(cfg,accelerator,buffer_size_seq,buffer_size_token)
        super().__init__(cfg, accelerator)
        self.max_steps = max_steps

    def __call__(self, batch, model_output, step, metrics_dict):

        self.step = step
        self.attention_mask = batch.get("attention_mask", None)
        self.batch = batch
        self.model_output = model_output
        self.gate_logits = model_output["router_logits"]
        self.logits = model_output["logits"]

        # print("test step:", step)
        # print("max steps:", self.max_steps)

        # if step%100==0:
        #     print("visualise")
        #     self.visualise_token_routing_paths(self.batch, self.gate_logits, metrics_dict)

        # #at each batch
        self.build_token_expert_usage_dic(
            self.batch["input_ids"], self.gate_logits, self.attention_mask, self.cfg
        )
        self.build_sequence_expert_usage_dic(
            self.batch["input_ids"], self.gate_logits, self.attention_mask, self.cfg
        )
        # self.accumulate_ce_and_expert_usage_seq(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch)
        # self.accumulate_ce_and_expert_usage_token(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch,self.tokenizer)

        self.accumulate_ce_and_expert_usage_seq(
            self.logits, self.gate_logits, self.attention_mask, self.cfg, self.batch
        )
        self.accumulate_ce_and_expert_usage_token(
            self.logits,
            self.gate_logits,
            self.attention_mask,
            self.cfg,
            self.batch,
            self.tokenizer,
        )
        self.log_correlation_between_ce_and_expert_usage_seq(metrics_dict)
        self.log_correlation_between_ce_and_expert_usage_token(
            metrics_dict, step=self.step
        )

        # #only at the end of training
        if step == self.max_steps - 5:  # we take a margin because of dynamic rebatching
            #     print("logging final analysis results to wandb")
            self.log_sequence_expert_usage_df(
                self.sequence_expert_usage_dict, metrics_dict
            )
            self.log_token_expert_usage_df(self.token_expert_usage_dict, metrics_dict)
        #     self.log_correlation_between_ce_and_expert_usage_seq(metrics_dict,prefix="test")
        #     self.log_correlation_between_ce_and_expert_usage_token(metrics_dict,prefix="test")

        self.accelerator.log(metrics_dict)
