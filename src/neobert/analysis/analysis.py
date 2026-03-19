import torch
from typing import Union
from torch.nn import CrossEntropyLoss
from omegaconf import DictConfig
from collections import defaultdict
import pandas as pd
from ..tokenizer import get_tokenizer
from .analysis_utils import AnalysisMetrics
import wandb


class AnalysisLogger(AnalysisMetrics):
    def __init__(self, cfg, accelerator, buffer_size_seq=1000, buffer_size_token=5000):
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
        # divide by 2 to be certain to fill the buffer in a single batch even if some tokens are padding tokens
        # wandb table initialiser
        self.wandb_sequence_expert_usage_table_initialized = False
        self.wandb_token_expert_usage_table_initialized = False

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

    def log_expert_usage_per_layer_plots(self):
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
            self.accelerator.log(
                {
                    "train/max_expert_usage_per_layer": wandb.plot.line_series(
                        xs=xs,
                        ys=ys,
                        keys=max_columns[1:],
                        title="Max Expert Usage Per Layer",
                        xname=max_columns[0],
                    )
                }
            )
        if hasattr(self, "mean_expert_usage_per_layer_df"):
            mean_columns = self.mean_expert_usage_per_layer_df.columns
            xs = self.mean_expert_usage_per_layer_df[mean_columns[0]].tolist()
            ys = [
                self.mean_expert_usage_per_layer_df[col].tolist()
                for col in mean_columns[1:]
            ]
            self.accelerator.log(
                {
                    "train/mean_expert_usage_per_layer": wandb.plot.line_series(
                        xs=xs,
                        ys=ys,
                        keys=mean_columns[1:],
                        title="Mean Expert Usage Per Layer",
                        xname=mean_columns[0],
                    )
                }
            )

    def log_and_append_layer_level_metrics(
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
        mean_expert_usage_per_layer_per_token, _ = (
            self._get_mean__and_max_expert_usage_per_layer_per_token(
                gate_logits, attention_mask, cfg
            )
        )
        self.difference_mean_expert_usage_across_layers = (
            self.get_difference_mean_expert_usage_across_layers(
                mean_expert_usage_per_layer_per_token
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
        # Correlation metric
        self.accelerator.log(
            {
                "train/difference_mean_expert_usage_across_layers": self.difference_mean_expert_usage_across_layers.item()
            }
        )

    def log_expert_usage_proportions_plots(self):
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
            self.accelerator.log(
                {
                    "train/expert_usage_proportions": wandb.plot.line_series(
                        xs=xs,
                        ys=ys,
                        keys=prop_columns[1:],
                        title="Expert Usage Proportions",
                        xname=prop_columns[0],
                    )
                }
            )

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

    def log_correlation_between_ce_and_expert_usage_token(
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

        # When buffer is full, compute correlation and log, then reset
        if (
            len(self.expert_usage_token_buffer) >= self.buffer_size_token
            and len(self.ce_loss_token_buffer) >= self.buffer_size_token
        ):
            # Truncate to buffer_size in case of overflow
            expert_usage_arr = torch.tensor(
                self.expert_usage_token_buffer[: self.buffer_size_token]
            )
            cross_entropy_loss_arr = torch.tensor(
                self.ce_loss_token_buffer[: self.buffer_size_token]
            )
            # Compute Pearson correlation
            if expert_usage_arr.std() > 0 and cross_entropy_loss_arr.std() > 0:
                correlation = torch.corrcoef(
                    torch.stack([expert_usage_arr, cross_entropy_loss_arr])
                )[0, 1].item()
            else:
                correlation = 0.0
            # Log correlation
            self.accelerator.log(
                {"train/expert_usage_cross_entropy_per_token_corr": correlation}
            )
            # Reset buffers
            self.expert_usage_token_buffer = []
            self.ce_loss_token_buffer = []

    def log_correlation_between_ce_and_expert_usage_seq(
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
            if expert_usage_arr.std() > 0 and cross_entropy_loss_arr.std() > 0:
                correlation = torch.corrcoef(
                    torch.stack([expert_usage_arr, cross_entropy_loss_arr])
                )[0, 1].item()
            else:
                correlation = 0.0
            # Log correlation
            self.accelerator.log(
                {"train/expert_usage_cross_entropy_per_seq_corr": correlation}
            )
            # Reset buffers
            self.expert_usage_seq_buffer = []
            self.ce_loss_seq_buffer = []

    def log_entropy_metrics(
        self,
        gate_logits: tuple[torch.Tensor],
        attention_mask: torch.Tensor,
        cfg: DictConfig,
    ):

        self.entropy = self.get_entropy(gate_logits, attention_mask, cfg)
        self.entropy_per_sequence = self.get_entropy_per_sequence(
            gate_logits, attention_mask, cfg
        )

        self.accelerator.log(
            {"train/entropy_per_sequence": self.entropy_per_sequence.item()}
        )
        self.accelerator.log({"train/entropy": self.entropy.item()})

    def log_std_rel_metrics(
        self,
        logits: torch.Tensor,
        gate_logits: tuple[torch.Tensor],
        attention_mask: torch.Tensor,
        cfg: DictConfig,
    ):
        # metrics
        self.intra_sequence_normalised_expert_cost_rel_std = (
            self.get_intra_sequence_normalised_expert_cost_rel_std(
                gate_logits, attention_mask, cfg
            )
        )
        self.inter_token_normalised_expert_cost_rel_std = (
            self.get_inter_token_normalised_expert_cost_rel_std(
                gate_logits, attention_mask, cfg
            )
        )
        self.inter_sequence_normalised_expert_cost_rel_std = (
            self.get_inter_sequence_normalised_expert_cost_rel_std(
                gate_logits, attention_mask, cfg
            )
        )

        # log metrics
        self.accelerator.log(
            {
                "train/intra_sequence_normalised_expert_cost_rel_std": self.intra_sequence_normalised_expert_cost_rel_std.item(),
                "train/inter_token_normalised_expert_cost_rel_std": self.inter_token_normalised_expert_cost_rel_std.item(),
                "train/inter_sequence_normalised_expert_cost_rel_std": self.inter_sequence_normalised_expert_cost_rel_std.item(),
            }
        )

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

    def log_token_expert_usage_df(self, token_expert_usage_dict: defaultdict):
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

        self.accelerator.log({"token_expert_usage": self.token_expert_usage_table})

    def log_sequence_expert_usage_df(self, sequence_expert_usage_dict: defaultdict):
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

        self.accelerator.log(
            {"sequence_expert_usage": self.sequence_expert_usage_table}
        )


class AnalysisTraining(AnalysisLogger):
    def __init__(
        self,
        cfg,
        accelerator,
    ):
        super().__init__(cfg, accelerator)

    def __call__(self, batch, model_output, step):

        self.step = step
        self.attention_mask = batch.get("attention_mask", None)
        self.batch = batch
        self.model_output = model_output
        self.gate_logits = model_output["router_logits"]
        self.logits = model_output["logits"]

        # at each batch
        # self.build_token_expert_usage_dic(self.batch["input_ids"], self.gate_logits, self.attention_mask, self.cfg)
        # self.build_sequence_expert_usage_dic(self.batch["input_ids"], self.gate_logits, self.attention_mask, self.cfg)
        self.log_correlation_between_ce_and_expert_usage_seq(
            self.logits, self.gate_logits, self.attention_mask, self.cfg, self.batch
        )
        self.log_correlation_between_ce_and_expert_usage_token(
            self.logits,
            self.gate_logits,
            self.attention_mask,
            self.cfg,
            self.batch,
            self.tokenizer,
        )

        # at each log step
        if step % self.cfg.wandb.log_interval == 0:
            self.log_and_append_layer_level_metrics(
                self.gate_logits, self.attention_mask, self.cfg, self.step
            )
            self.append_expert_level_metrics(
                self.gate_logits, self.attention_mask, self.cfg, self.step
            )
            self.log_entropy_metrics(self.gate_logits, self.attention_mask, self.cfg)
            self.log_std_rel_metrics(
                self.logits, self.gate_logits, self.attention_mask, self.cfg
            )

            self.build_token_expert_usage_dic(
                self.batch["input_ids"], self.gate_logits, self.attention_mask, self.cfg
            )
            self.build_sequence_expert_usage_dic(
                self.batch["input_ids"], self.gate_logits, self.attention_mask, self.cfg
            )
            # self.log_correlation_between_ce_and_expert_usage_seq(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch)
            # self.log_correlation_between_ce_and_expert_usage_seq(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch)
            # self.log_correlation_between_ce_and_expert_usage_token(self.logits,self.gate_logits, self.attention_mask, self.cfg,self.batch,self.tokenizer)

        # only at the end of training
        if step == self.cfg.trainer.max_steps - 1:
            print("logging final analysis results to wandb")
            self.log_sequence_expert_usage_df(self.sequence_expert_usage_dict)
            self.log_token_expert_usage_df(self.token_expert_usage_dict)
            self.log_expert_usage_per_layer_plots()
            self.log_expert_usage_proportions_plots()


class AnalysisTrainedModel(AnalysisLogger):
    def __init__(
        self,
        cfg,
        accelerator,
    ):
        super().__init__(cfg, accelerator)

    def __call__(self, batch, model_output, step):

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
        self.log_correlation_between_ce_and_expert_usage_seq(
            self.logits, self.gate_logits, self.attention_mask, self.cfg, self.batch
        )
        self.log_correlation_between_ce_and_expert_usage_token(
            self.logits,
            self.gate_logits,
            self.attention_mask,
            self.cfg,
            self.batch,
            self.tokenizer,
        )

        self.log_and_append_layer_level_metrics(
            self.gate_logits, self.attention_mask, self.cfg, self.step
        )
        self.append_expert_level_metrics(
            self.gate_logits, self.attention_mask, self.cfg, self.step
        )
        self.log_entropy_metrics(self.gate_logits, self.attention_mask, self.cfg)
        self.log_std_rel_metrics(
            self.logits, self.gate_logits, self.attention_mask, self.cfg
        )

        # at each log step
        if step % self.cfg.wandb.log_interval == 0:
            self.log_and_append_layer_level_metrics(
                self.gate_logits, self.attention_mask, self.cfg, self.step
            )
            self.append_expert_level_metrics(
                self.gate_logits, self.attention_mask, self.cfg, self.step
            )
            self.log_entropy_metrics(self.gate_logits, self.attention_mask, self.cfg)
            self.log_std_rel_metrics(
                self.logits, self.gate_logits, self.attention_mask, self.cfg
            )

        # only at the end of training
        if step == self.cfg.trainer.max_steps - 1:
            print("logging final analysis results to wandb")
            self.log_sequence_expert_usage_df(self.sequence_expert_usage_dict)
            self.log_token_expert_usage_df(self.token_expert_usage_dict)
            self.log_expert_usage_per_layer_plots()
            self.log_expert_usage_proportions_plots()
