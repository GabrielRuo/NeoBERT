
import torch
from typing import Union, Optional, Any, Dict
from torch.nn import CrossEntropyLoss

def load_balancing_loss_fn(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        attention_mask= torch.where(attention_mask == float(0.0), 1, float(0.0)) #original mask is additive-->1,0
        # tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # # Compute the average probability of routing to these experts
        # router_prob_per_expert = torch.mean(routing_weights, dim=0)
        
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

def hmoe_penalty_loss_fn(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    cfg: Dict[str, Any],
    num_experts: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        cfg:
            The configuration object
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    expert_sizes = torch.tensor([int(size) for size in cfg.model.expert_sizes.split(",")], device=compute_device)

    if cfg.model.routing_strategy == "top_k":
        top_k = cfg.model.num_experts_per_tok #if we wanted  to use top_k for heterogeneous moe
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)# n_tokens, top_k, num_experts
        expert_mask = torch.sum(expert_mask, dim=1) * expert_sizes  # n_tokens, num_experts
        mean_num_activated_experts = top_k

    elif cfg.model.routing_strategy == "top_p":
        top_p = cfg.model.min_expert_cumprob_per_token
        sorted_weights, sorted_indices = torch.sort(routing_weights, dim=-1, descending=False)

        cum_probs = sorted_weights.cumsum(dim=-1)
        mask = cum_probs > 1 - top_p

        unsorted_mask = torch.zeros_like(mask, dtype=torch.bool)
        expert_mask = unsorted_mask.scatter(dim=-1, index=sorted_indices, src=mask).float()#n_layers*batch_size*seq_len, num_experts
        mean_num_activated_experts = torch.mean(torch.sum(expert_mask.float(), dim=1), dim=0)
        expert_mask *= expert_sizes # n_tokens, num_experts


    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        # tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # # Compute the average probability of routing to these experts
        # router_prob_per_expert = torch.mean(routing_weights, dim=0)

        attention_mask= torch.where(attention_mask == float(0.0), 1, float(0.0)) #original mask is additive-->1,0

        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :,None]
            .expand((num_hidden_layers, batch_size, sequence_length,num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )
        
        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts, mean_num_activated_experts

def hmoe_entropy_loss_fn(gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None], cfg, attention_mask):

    #ATTENTION: I WILL NEED TO ADAPT THIS TO THE CASE WHERE THERE IS AN ATTENTION MASK
    """
    Computes the entropy loss for the Heterogeneous Mixture of Experts (HMoE) model.

    Args:
        gate_logits: Logits from the gate, should be a tensor of shape [batch_size, sequence_length, num_experts].
        cfg: Configuration object containing model parameters.
        batch: The input batch containing labels and other necessary information.

    Returns:
        The computed entropy loss.
    """
    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        if attention_mask is None:
            concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)#n_layers*batch_size*seq_length,num_expert
        else:
            concatenated_gate_logits = torch.stack([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)#n_layers,batch_size*seq_length,num_experts
            boolean_pad_mask = (attention_mask.reshape(-1) != float("-inf")).squeeze(-1)#[batch_size*seq_length,num_experts]
            concatenated_gate_logits = concatenated_gate_logits[:,boolean_pad_mask,:].reshape(-1, concatenated_gate_logits.shape[-1]) #n_layers*batch_size*seq_length,num_expert

    else: 
        print("WARNING: gate_logits is not a tuple, this is not supported in the current implementation of entropy loss function. ")

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)


    # Compute the entropy
    entropy_loss = -torch.sum(routing_weights * torch.log(routing_weights + cfg.model.loss.entropy_loss_epsilon), dim=-1)

    # Average
    entropy_loss = torch.mean(entropy_loss)

    return entropy_loss 

def hetero_moe_loss_fn(logits, router_logits, cfg, batch):
    """
    Computes the loss for the Heterogeneous Mixture of Experts (HMoE) model.

    Args:
        logits: The output logits from the model.
        router_logits: The logits from the router.
        cfg: Configuration object containing model parameters.
        batch: The input batch containing labels and other necessary information.

    Returns:
        The computed loss.
    """
    num_experts = len(cfg.model.expert_sizes.split(','))

    mlm_loss_fn = CrossEntropyLoss()
    mlm_loss = mlm_loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), batch["labels"].view(-1))

    penalty_loss,mean_num_activated_experts = hmoe_penalty_loss_fn(router_logits, cfg, num_experts=num_experts, attention_mask=batch.get("attention_mask", None))

    entropy_loss = hmoe_entropy_loss_fn(router_logits,cfg, attention_mask=batch.get("attention_mask", None))

    train_loss = mlm_loss + cfg.model.loss.penalty_loss_coeff * penalty_loss + cfg.model.loss.entropy_loss_coeff * entropy_loss

    return train_loss,mlm_loss,penalty_loss,entropy_loss, mean_num_activated_experts


def homo_moe_loss_fn(logits,router_logits,cfg, batch):

    num_experts = len(cfg.model.expert_sizes.split(','))
    top_k = cfg.model.num_experts_per_tok_training

    mlm_loss_fn = CrossEntropyLoss()
    mlm_loss = mlm_loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), batch["labels"].view(-1))
    load_balancing_loss = load_balancing_loss_fn(router_logits, top_k=top_k, num_experts=num_experts, attention_mask=batch.get("attention_mask", None))

    train_loss = mlm_loss + cfg.model.loss.load_balancing_loss_coeff * load_balancing_loss

    return train_loss,mlm_loss,load_balancing_loss



def mop_loss_fn(logits, expert_usage_loss, cfg, batch, num_tokens):
    mlm_loss_fn = CrossEntropyLoss()

    train_loss = masked_lm_loss = mlm_loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), batch["labels"].view(-1))#reshape since cross entropy expects (n_tokens, n_classes)

    expert_dims = [int(expert_size) for expert_size in cfg.model.expert_sizes.split(",")]
    normalisation_factor = sum(expert_dims)


    if cfg.model.loss.cost_based_loss_alpha_start > 0:
            
            #compute logsum then exp to prevent overflow
            #cost_based_loss_alpha = min(cfg.model.loss.cost_based_loss_alpha_end, cfg.model.loss.cost_based_loss_alpha_start+ (cfg.model.loss.cost_based_loss_alpha_end - cfg.model.loss.cost_based_loss_alpha_start)*num_tokens / cfg.model.loss.cost_based_loss_schedule_tokens)
            cost_based_loss_alpha = cfg.model.loss.cost_based_loss_alpha_end if num_tokens>cfg.model.loss.cost_based_loss_schedule_tokens else cfg.model.loss.cost_based_loss_alpha_start
            cost_based_loss_alpha = torch.tensor(cost_based_loss_alpha, dtype=torch.float32, device=expert_usage_loss.device)
           
            normalisation_factor = torch.tensor(normalisation_factor, dtype=torch.float32, device=expert_usage_loss.device)
            logsum = cfg.model.expert_cost_exponent*torch.log(normalisation_factor)+torch.log(cost_based_loss_alpha)+torch.log(expert_usage_loss)
            loss_numerator =torch.exp(logsum).to(dtype=expert_usage_loss.dtype)
        
            if cfg.model.loss.disable_task_performance_scaling:
                expert_loss = loss_numerator
            else:
                loss_denominator = cfg.model.loss.cost_based_loss_epsilon + masked_lm_loss
                expert_loss = loss_numerator / loss_denominator

            train_loss = masked_lm_loss + expert_loss

    return train_loss, masked_lm_loss, expert_loss, cost_based_loss_alpha

from .analysis import get_normalised_expert_usage_cost_per_sequence

def mop_loss_fn_alt(logits, router_logits, cfg, batch, num_tokens):

    #compute a separate mlm_loss for each sequence in the batch
    mlm_loss_fn = CrossEntropyLoss(reduction = "none")
    train_loss = masked_lm_loss = mlm_loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), batch["labels"].view(-1))#shape seq_length*nbatch_size
    masked_lm_loss = masked_lm_loss.reshape(logits.size(0), logits.size(1))#shape (batch_size, seq_length)
    mask = (batch["labels"] != -100).float()  # (batch_size, seq_len)
    # Sum losses and normalize by number of valid tokens in each sequence
    masked_lm_loss = (masked_lm_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)#shape (batch_size,)

    #get the expert loss per sequence
    expert_usage_loss = get_normalised_expert_usage_cost_per_sequence(router_logits, batch.get("attention_mask", None), cfg)#shape (batch_size,)

    expert_dims = [int(expert_size) for expert_size in cfg.model.expert_sizes.split(",")]
    normalisation_factor = sum(expert_dims)


    if cfg.model.loss.cost_based_loss_alpha_start > 0:
            
            #compute logsum then exp to prevent overflow
            #cost_based_loss_alpha = min(cfg.model.loss.cost_based_loss_alpha_end, cfg.model.loss.cost_based_loss_alpha_start+ (cfg.model.loss.cost_based_loss_alpha_end - cfg.model.loss.cost_based_loss_alpha_start)*num_tokens / cfg.model.loss.cost_based_loss_schedule_tokens)
            cost_based_loss_alpha = cfg.model.loss.cost_based_loss_alpha_end if num_tokens>cfg.model.loss.cost_based_loss_schedule_tokens else cfg.model.loss.cost_based_loss_alpha_start
            cost_based_loss_alpha = torch.tensor(cost_based_loss_alpha, dtype=torch.float32, device=expert_usage_loss.device)
           
            normalisation_factor = torch.tensor(normalisation_factor, dtype=torch.float32, device=expert_usage_loss.device)
            logsum = cfg.model.expert_cost_exponent*torch.log(normalisation_factor)+torch.log(cost_based_loss_alpha)+torch.log(expert_usage_loss)
            loss_numerator =torch.exp(logsum).to(dtype=expert_usage_loss.dtype)#shape (batch_size,)

            if cfg.model.loss.disable_task_performance_scaling:
                expert_loss = loss_numerator
            else:
                loss_denominator = cfg.model.loss.cost_based_loss_epsilon + masked_lm_loss
                expert_loss = loss_numerator / loss_denominator

            train_loss = masked_lm_loss + expert_loss
            train_loss = torch.mean(train_loss)

    return train_loss, masked_lm_loss.mean(), expert_loss.mean(), cost_based_loss_alpha