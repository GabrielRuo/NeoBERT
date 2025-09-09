import torch
from typing import Union
from torch.nn import CrossEntropyLoss

def get_entropy(gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None], cfg, attention_mask):

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

    entropy_loss_epsilon = 1e-5
    
    # Compute the entropy
    entropy_loss = -torch.sum(routing_weights * torch.log(routing_weights + entropy_loss_epsilon), dim=-1)

    # Average
    entropy_loss = torch.mean(entropy_loss)

    return entropy_loss

def get_normalised_expert_usage_cost_per_sequence(router_logits, attention_mask,cfg):
        #compute the expert usage cost per sequence in a batch
        #keep it normalised by the  size of the experts as we do not care about the absolute value. This will make the valus fairly small.

        # Correct stacking of tuple of raw_routing_weights            
        router_logits = torch.stack(list(router_logits), dim=0) #shape (n_layers, batch * sequence_length, n_experts)
        raw_routing_weights = torch.softmax(router_logits, dim=-1)

        if len(cfg.model.expert_sizes.split(',')) > 1:
            expert_dims = [int(size) for size in cfg.model.expert_sizes.split(',')]
            expert_dims_tensor = torch.tensor(expert_dims,dtype=torch.float32,
                device=raw_routing_weights.device )
            normalisation_factor = torch.sum(expert_dims_tensor)
            normalised_expert_sizes = expert_dims_tensor / normalisation_factor#prevent overflow
            routing_costs = normalised_expert_sizes**cfg.model.expert_cost_exponent
            routing_costs = routing_costs.to(dtype = raw_routing_weights.dtype)#cast back to original dtype

            n_layers,_,n_experts = raw_routing_weights.shape
            batch_size, seq_length = attention_mask.shape
            raw_routing_weights = raw_routing_weights.reshape(n_layers, batch_size, seq_length, n_experts)
            number_of_tokens_per_seq = seq_length
            

            #ignore padding_tokens
            if attention_mask!= None: #attention_mask has shape (Batch,seq_length) and routing_weights has shape (n_layers, batch * sequence_length, n_experts)
                multiplicative_mask = torch.where(attention_mask == 0, 1.0, 0.0).to(dtype = raw_routing_weights.dtype).reshape(1, batch_size, seq_length, 1)
                number_of_tokens_per_seq = multiplicative_mask.sum(dim=2).squeeze()
                raw_routing_weights = raw_routing_weights * multiplicative_mask
            # Compute usage loss per token
            expert_usage_loss = torch.einsum('mijk,k->mij', raw_routing_weights, routing_costs) #(n_layers,batch_size,seq_length)
            
            #sum across layers
            expert_usage_loss = torch.sum(expert_usage_loss, dim=0) #shape (batch_size,seq_length)
            # Average over all tokens in the sequence
            normalised_mean_expert_usage_loss = torch.sum(expert_usage_loss, dim = 1) / number_of_tokens_per_seq #shape (batch_size,)
        else:
            normalised_mean_expert_usage_loss = None

        return normalised_mean_expert_usage_loss #shape (batch_size,) or None if single expert

def get_mse_per_sequence(logits, cfg, batch):
    """
    Compute the Mean Squared Error (MSE) loss per sequence.

    Args:
        logits: The model predictions. #shape (batch_size, seq_length, vocab_size)
        cfg: Configuration object containing model parameters.
        labels: The ground truth labels. #shape (batch_size, seq_length)

    Returns:
        The computed MSE loss per sequence.
        
    """
    mlm_loss_fn = CrossEntropyLoss(reduction = "none")
    train_loss = masked_lm_loss = mlm_loss_fn(logits.view(-1, cfg.tokenizer.vocab_size), batch["labels"].view(-1))#shape seq_length*nbatch_size
    masked_lm_loss = masked_lm_loss.reshape(logits.size(0), logits.size(1))#shape (batch_size, seq_length)
    mask = (batch["labels"] != -100).float()  # (batch_size, seq_len)
    # Sum losses and normalize by number of valid tokens in each sequence
    masked_lm_loss = (masked_lm_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)#shape (batch_size,)

    return masked_lm_loss #shape (batch_size,): average loss per sequence ignoring tokens not used fo loss computation
