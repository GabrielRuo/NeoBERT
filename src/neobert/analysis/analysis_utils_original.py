import torch
from typing import Union
from torch.nn import CrossEntropyLoss
from omegaconf import DictConfig
from collections import defaultdict
import pandas as pd
from ..tokenizer import get_tokenizer
import numpy as np

class AnalysisMetrics:
    
    def _get_concatenated_gate_logits(self,gate_logits: tuple[torch.Tensor], attention_mask: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
        """
        Concatenate the gate logits from all layers into a single tensor.

        Args:
            gate_logits (tuple[torch.Tensor]): Logits from the gate, should be a tuple of n_layers tensors of shape [batch_size*sequence_length, num_experts].
            attention_mask (torch.Tensor): The additive attention mask for the input sequences (0 for tokens to attend to, -inf for padding tokens): shape [batch_size, sequence_length].

        Returns:
            torch.Tensor: Concatenated gate logits of shape [n_layers*batch_size*sequence_length, num_experts].
        """
        assert isinstance(gate_logits, tuple), "gate_logits should be a tuple of tensors, one per layer"

        n_layers = len(gate_logits)
        batch_size = cfg.dataloader.train.batch_size
        seq_length = cfg.tokenizer.max_length
        compute_device = gate_logits[0].device

        # Convert tuple to a single tensor
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)  # n_layers*batch_size*seq_length,num_experts

        # Apply attention mask to ignore padding tokens
        concatenated_gate_logits = concatenated_gate_logits.reshape(n_layers, batch_size, seq_length, -1)  # shape (n_layers,batch_size, seq_length,num_experts)
        if attention_mask is not None:
            multiplicative_attention_mask = (attention_mask != float('-inf')).float().to(compute_device)  # 1 for unmasked tokens, 0 for masked tokens
            concatenated_gate_logits = concatenated_gate_logits * multiplicative_attention_mask.reshape(1, batch_size, seq_length, 1)  # shape (n_layers,batch_size,seq_length,num_experts)
        
        return concatenated_gate_logits

    def get_entropy(self, gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None], attention_mask: torch.Tensor, entropy_loss_epsilon: float = 1e-5) -> torch.Tensor:

        """
        Computes the entropy loss for the Heterogeneous Mixture of Experts (HMoE) model.

        Args:
            gate_logits: Logits from the gate, should be a tuple of n_layers tensors of shape [batch_size*sequence_length, num_experts].
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

        # Compute the entropy of the distibution of each token of every layer
        entropy_loss = -torch.sum(routing_weights * torch.log(routing_weights + entropy_loss_epsilon), dim=-1)

        # Average across all tokens in the batch -> average entropy per token
        entropy_loss = torch.mean(entropy_loss)

        return entropy_loss

    #layer level metrics
    def get_correlation_mean_expert_usage_across_layers(self,mean_expert_usage_per_layer: torch.Tensor) -> torch.Tensor:
        """Compute the correlation between the mean expert between layer N and layer N+1 for a given token across all tokens and all N,N+1 pairs.
        A clear negative correlation would show that when a token has been routed to a complex expert it is now considered "more simple": this would make routing predictable


        Args:
            mean_expert_usage_per_layer (torch.Tensor): shape [n_layers,n_seq_length*batch_size*number_of_batches]
        """
        n_layers = mean_expert_usage_per_layer.shape[0]
        correlations = []
        for i in range(n_layers - 1):
            layer_n = mean_expert_usage_per_layer[i]
            layer_n_plus_1 = mean_expert_usage_per_layer[i + 1]
            if torch.std(layer_n) > 0 and torch.std(layer_n_plus_1) > 0:
                correlation = torch.corrcoef(torch.stack([layer_n, layer_n_plus_1]))[0, 1]
                correlations.append(correlation)
        if len(correlations) > 0:
            mean_correlation = torch.mean(torch.stack(correlations))
        else:
            mean_correlation = torch.tensor(0.0, device=mean_expert_usage_per_layer.device)
        return mean_correlation


    def get_mean_and_max_expert_usage_per_layer(self,gate_logits: tuple[torch.Tensor], attention_mask: torch.Tensor, cfg: DictConfig) -> tuple[torch.Tensor,torch.Tensor]:
        """
        Compute the mean expert selected at each layer averaged across all tokens in the batch.
        This tells us whether different layers tend to use different experts on average.

        Args:
            gate_logits (tuple[torch.Tensor]): Logits from the gate, should be a tuple of n_layers tensors of shape [batch_size*sequence_length, num_experts].
            attention_mask (torch.Tensor): The additive attention mask for the input sequences (0 for tokens to attend to, -inf for padding tokens): shape [batch_size, sequence_length].

        Returns:
            torch.Tensor: The distribution of logits sent to each expert across all tokens in the batch.
        """
        mean_selected_expert_per_layer_per_token, max_selected_expert_per_layer_per_token = self._get_mean__and_max_expert_usage_per_layer_per_token(gate_logits, attention_mask, cfg)  # shape (n_layers,batch_size*seq_length)
        mean_selected_expert_per_layer = torch.mean(mean_selected_expert_per_layer_per_token, dim=1)  # shape (n_layers,) #mean across tokens
        max_selected_expert_per_layer = torch.max(max_selected_expert_per_layer_per_token, dim=1)[0]  # shape (n_layers,) #max across tokens

        return mean_selected_expert_per_layer, max_selected_expert_per_layer


    def _get_mean__and_max_expert_usage_per_layer_per_token(self,gate_logits: tuple[torch.Tensor], attention_mask: torch.Tensor, cfg: DictConfig) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(gate_logits, tuple), "gate_logits should be a tuple of tensors, one per layer"

        n_layers = len(gate_logits)
        num_experts = gate_logits[0].shape[-1]
        batch_size = cfg.dataloader.train.batch_size
        seq_length = cfg.tokenizer.max_length
        compute_device = gate_logits[0].device
        expert_indices = torch.arange(num_experts, device=compute_device, dtype=gate_logits[0].dtype)

        # Convert tuple to a single tensor
        concatenated_gate_logits = self._get_concatenated_gate_logits(gate_logits, attention_mask, cfg)  # n_layers*batch_size*seq_length,num_experts

        # Aggregate the probabilities per expert across all tokens
        concatenated_gate_logits_per_layer = concatenated_gate_logits.reshape(n_layers,-1, num_experts)  # shape (n_layers,batch_size*seq_length,num_experts)
        concatenated_gate_softmax_per_layer = torch.nn.functional.softmax(concatenated_gate_logits_per_layer, dim=-1) # shape (n_layers,batch_size*seq_length,num_experts)
        mean_selected_expert_per_layer = torch.einsum('men,n->me', concatenated_gate_softmax_per_layer, expert_indices)  # shape (n_layers,batch_size*seq_length)
        max_selected_expert_per_layer = torch.argmax(concatenated_gate_softmax_per_layer, dim=-1)  # shape (n_layers,batch_size*seq_length)

        return mean_selected_expert_per_layer,max_selected_expert_per_layer

    #token level metrics
    #token_expert_usage_dict = defaultdict(lambda:[])

    def get_df_of_average_token_usage(self,token_expert_usage_dict: defaultdict,tokenizer) -> pd.DataFrame:
        """Convert a dictionary of token expert usage information into a pandas DataFrame.
        The stdrel tells us to what extent a token is routed differently depending on the context.
        """
        token_expert_usage_dict_avg = {token_id: sum(usages) / len(usages) for token_id, usages in token_expert_usage_dict.items()}
        token_expert_usage_dict_rel_std = {token_id: np.sqrt(np.sum((np.array(usages) - token_expert_usage_dict_avg[token_id]) ** 2) / len(usages)) / token_expert_usage_dict_avg[token_id] for token_id, usages in token_expert_usage_dict.items()}  #sort by average expert usage
        #convert this  to a panda  data frame with columns token_id, avg_expert_usage, rel_std_expert_usage
        df = pd.DataFrame({
            'token_id': list(token_expert_usage_dict_avg.keys()),
            'written_token': [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in token_expert_usage_dict_avg.keys()],
            'avg_expert_usage': list(token_expert_usage_dict_avg.values()),

            'rel_std_expert_usage': list(token_expert_usage_dict_rel_std.values())
        })
        df = df.sort_values(by='avg_expert_usage', ascending=False)
        return df


    def _append_expert_usage_per_token(self,token_expert_usage_dict: defaultdict, batch_input_id:torch.Tensor, batch_gate_logits: tuple[torch.Tensor], batch_attention_mask: torch.Tensor, cfg: DictConfig) -> defaultdict:
        """Append expert usage information for each token in the batch.
        This function builds a given dictionary where the key is the token_id and the value is a list of expert usage values for that token across different occurrences in the batch.
        Expert usage is defined as the mean expert index selected across all layers for a given token. In the end we really care about how routing is going on, not the expert loss
        (1)This can be used to verify whether at a given training points, a given token follows a consistant routing pattern.
        (2)Taking averages across all occurrences of a token, it can help rank tokens by complexity
        (3)Evolution of token complexity throughout  training: if the relative diff between token complexity and mean complexity diminishes with number of occurrences, then model 
        learns to make token simpler as it sees  it more often. This is similar to correlating mse with expert complexity

        Args:
            token_expert_usage_dict (defaultdict(lambda:[]): Dictionary to store expert usage information (key: token_id, value: list of expert usage values).
            batch_input_id (torch.Tensor): Input IDs for the batch. shape (batch_size, seq_length)  
            batch_gate_logits (tuple[torch.Tensor]): Gate logits for the batch. Should be a tuple of n_layers tensors of shape [batch_size*sequence_length, num_experts].
        """
        n_layers = len(batch_gate_logits)
        num_experts = batch_gate_logits[0].shape[-1]
        batch_size = cfg.dataloader.train.batch_size
        seq_length = cfg.tokenizer.max_length
        compute_device = batch_gate_logits[0].device
        expert_indices = torch.arange(num_experts, device=compute_device, dtype=batch_gate_logits[0].dtype)

        batch_concatenated_gate_logits = self._get_concatenated_gate_logits(batch_gate_logits, batch_attention_mask, cfg)#shape (n_layers,batch_size, seq_length,num_experts)
        batch_concatenated_gate_softmax = torch.nn.functional.softmax(batch_concatenated_gate_logits, dim=-1) #shape (n_layers,batch_size, seq_length,num_experts)
        mean_selected_expert_per_token =  torch.einsum('nbse,e->nbs', batch_concatenated_gate_softmax, expert_indices)# shape (n_layers,batch_size, seq_length))
        mean_expert_usage_per_token = torch.mean(mean_selected_expert_per_token, dim=0)  # shape (batch_size, seq_length): mean across layers
        mean_expert_usage_per_token = mean_expert_usage_per_token.reshape(-1)  # shape (batch_size*seq_length,)
        
        for token_index in range(batch_size * seq_length):
            token_id = batch_input_id.reshape(-1)[token_index].item() #why this?
            expert_usage = mean_expert_usage_per_token[token_index].item()
            token_expert_usage_dict[token_id].append(expert_usage)
        
        return token_expert_usage_dict


    #expert level metrics
    def get_expert_usage_cum_prob(self,gate_logits: tuple[torch.Tensor], attention_mask: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
        """
        Compute the distribution of routing weights sent to each expert across all tokens in the batch and layers.
        This tells us  how much each expert is used in absolute terms and we can plot this throughout training
        This is in absolute value, so needs to be normalised to get proportions.

        Args:
            gate_logits (tuple[torch.Tensor]): Logits from the gate, should be a tuple of n_layers tensors of shape [batch_size*sequence_length, num_experts].
            attention_mask (torch.Tensor): The additive attention mask for the input sequences (0 for tokens to attend to, -inf for padding tokens): shape [batch_size, sequence_length].

        Returns:
            torch.Tensor: The distribution of logits sent to each expert across all tokens in the batch.
        """
        assert isinstance(gate_logits, tuple), "gate_logits should be a tuple of tensors, one per layer"

        n_layers = len(gate_logits)
        num_experts = gate_logits[0].shape[-1]
        batch_size = cfg.dataloader.train.batch_size
        seq_length = cfg.tokenizer.max_length
        compute_device = gate_logits[0].device

        # Convert tuple to a single tensor
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)  # n_layers*batch_size*seq_length,num_experts

        # Apply attention mask to ignore padding tokens
        concatenated_gate_logits = concatenated_gate_logits.reshape(n_layers, batch_size, seq_length, num_experts)  # shape (n_layers,batch_size, seq_length,num_experts)
        if attention_mask is not None:
            multiplicative_attention_mask = (attention_mask != float('-inf')).float().to(compute_device)  # 1 for unmasked tokens, 0 for masked tokens
            concatenated_gate_logits = concatenated_gate_logits * multiplicative_attention_mask.reshape(1, batch_size, seq_length, 1)  # shape (n_layers,batch_size,seq_length,num_experts)

        # Aggregate the logits per expert across all tokens
        concatenated_gate_logits = concatenated_gate_logits.reshape(-1, seq_length, num_experts)  # shape (n_layers*batch_size, seq_length,num_experts)
        concatenated_gate_softmax = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)  # shape (n_layers*batch_size, seq_length,num_experts)
        expert_cum_prob = torch.sum(concatenated_gate_softmax, dim=(0,1))  # shape (n_layers*batch_size, num_experts)

        return expert_cum_prob


    #Sequence level metrics

    def get_df_of_average_sequence_usage(self,sequence_expert_usage_dict: defaultdict,tokenizer) -> pd.DataFrame:
        """Convert a dictionary of token expert usage information into a pandas DataFrame.
        The stdrel tells us to what extent a token is routed differently depending on the context.
        """
        sequence_expert_usage_dict_avg = {token_id: sum(usages) / len(usages) for token_id, usages in sequence_expert_usage_dict.items()}
        #convert this  to a panda  data frame with columns seq_id, avg_expert_usage, rel_std_expert_usage
        df = pd.DataFrame({
            'seq_id': list(sequence_expert_usage_dict_avg.keys()),
            'written_seq': [tokenizer.decode(list(token_id), skip_special_tokens=True) for token_id in sequence_expert_usage_dict_avg.keys()],
            'avg_expert_usage': list(sequence_expert_usage_dict_avg.values())
        })
        df = df.sort_values(by='avg_expert_usage', ascending=False)
        return df

    def _append_expert_usage_per_sequence(self,sequence_expert_usage_dict: defaultdict, batch_input_id:torch.Tensor, batch_gate_logits: tuple[torch.Tensor], batch_attention_mask: torch.Tensor, cfg: DictConfig) -> defaultdict:
        """Append expert usage information for each token in the batch.
        This function builds a given dictionary where the key is the token_id and the value is a list of expert usage values for that token across different occurrences in the batch.
        Expert usage is defined as the mean expert index selected across all layers for a given token. In the end we really care about how routing is going on, not the expert loss
        (1)This can be used to verify whether at a given training points, a given token follows a consistant routing pattern.
        (2)Taking averages across all occurrences of a token, it can help rank tokens by complexity
        (3)Evolution of token complexity throughout  training: if the relative diff between token complexity and mean complexity diminishes with number of occurrences, then model 
        learns to make token simpler as it sees  it more often. This is similar to correlating mse with expert complexity

        Args:
            token_expert_usage_dict (defaultdict(lambda:[]): Dictionary to store expert usage information (key: token_id, value: list of expert usage values).
            batch_input_id (torch.Tensor): Input IDs for the batch. shape (batch_size, seq_length)  
            batch_gate_logits (tuple[torch.Tensor]): Gate logits for the batch. Should be a tuple of n_layers tensors of shape [batch_size*sequence_length, num_experts].
        """
        n_layers = len(batch_gate_logits)
        num_experts = batch_gate_logits[0].shape[-1]
        batch_size = cfg.dataloader.train.batch_size
        seq_length = cfg.tokenizer.max_length
        compute_device = batch_gate_logits[0].device
        expert_indices = torch.arange(num_experts, device=compute_device, dtype=batch_gate_logits[0].dtype)

        batch_concatenated_gate_logits = self._get_concatenated_gate_logits(batch_gate_logits, batch_attention_mask, cfg)#shape (n_layers,batch_size, seq_length,num_experts)
        batch_concatenated_gate_softmax = torch.nn.functional.softmax(batch_concatenated_gate_logits, dim=-1) #shape (n_layers,batch_size, seq_length,num_experts)
        mean_selected_expert_per_token =  torch.einsum('nbse,e->nbs', batch_concatenated_gate_softmax, expert_indices)# shape (n_layers,batch_size, seq_length))
        mean_expert_usage_per_token = torch.mean(mean_selected_expert_per_token, dim=(0,2))  # shape (batch_size): mean across layers and seq_length

        for seq_index in range(batch_size):
            sequence_ids = batch_input_id[seq_index, :].tolist()
            expert_usage = mean_expert_usage_per_token[seq_index].item()
            sequence_expert_usage_dict[tuple(sequence_ids)].append(expert_usage)

        return sequence_expert_usage_dict

    def get_entropy_per_sequence(self, gate_logits: tuple[torch.Tensor], attention_mask: torch.Tensor, cfg: DictConfig, entropy_loss_epsilon: float = 1e-5) -> torch.Tensor:
        """
        Computes the entropy loss per sequence for the Heterogeneous Mixture of Experts (HMoE) model.
        When compared to the per-token average entropy, tells us whether tokens from a given sequence are routed to a similar set of experts or not.

        Args:
            gate_logits: Logits from the gate, should be a tuple of n_layers tensors of shape [batch_size*sequence_length, num_experts].
            attention_mask: The additive attention mask for the input sequences (0 for tokens to attend to, -inf for padding tokens): shape [batch_size, sequence_length].
            entropy_loss_epsilon: A small constant to avoid log(0).

        Returns:
            The average entropy  per sequence (size 1 scalar).
        """
        assert isinstance(gate_logits, tuple), "gate_logits should be a tuple of tensors, one per layer"

        n_layers = len(gate_logits)
        num_experts = gate_logits[0].shape[-1]
        batch_size = cfg.dataloader.train.batch_size
        seq_length = cfg.tokenizer.max_length
        compute_device = gate_logits[0].device

        #convert tuple to a single tensor
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)#n_layers*batch_size*seq_length,num_experts    
        
        # Apply attention mask to ignore padding tokens
        concatenated_gate_logits = concatenated_gate_logits.reshape(n_layers,batch_size,seq_length,num_experts)#shape (n_layers,batch_size, seq_length,num_experts)
        if attention_mask is not None:
            multiplicative_attention_mask = (attention_mask != float('-inf')).float().to(compute_device) #1 for unmasked tokens, 0 for masked tokens
            concatenated_gate_logits = concatenated_gate_logits * multiplicative_attention_mask.reshape(1,batch_size,seq_length, 1) #shape (n_layers,batch_size,seq_length,num_experts)

        #aggregate the logits per sequence at at each layer (reason for doing this per layer: maybe at different layers a sequence is routed to a different subset of experts):
        concatenated_gate_logits = concatenated_gate_logits.reshape(-1,seq_length,num_experts)#shape (n_layers*batch_size, seq_length,num_experts)
        sequence_logits = torch.sum(concatenated_gate_logits, dim=1) #shape (n_layers*batch_size, num_experts)
        sequence_weights = torch.nn.functional.softmax(sequence_logits, dim=-1)#shape (n_layers*batch_size, num_experts)

        # Compute the entropy of the distibution of each sequence at each layer
        entropy_per_sequence = -torch.sum(sequence_weights * torch.log(sequence_weights + entropy_loss_epsilon), dim=-1) #shape (n_layers*batch_size)

        #compute the average entropy across all sequences and layers
        mean_entropy_per_sequence = torch.mean(entropy_per_sequence)
        return mean_entropy_per_sequence
    
    def _get_expert_usage_per_token_and_layer(self, router_logits: tuple[torch.Tensor], attention_mask: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
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
            expert_usage_loss_per_token_and_layer = torch.einsum('mijk,k->mij', raw_routing_weights, routing_costs) #(n_layers,batch_size,seq_length)



    def get_intra_sequence_normalised_expert_cost_rel_std(self, router_logits: torch.Tensor, attention_mask: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
        """_summary_
        Compute the relative standard deviation of the expert usage cost per sequence in a batch: std/mean
        A high standard deviation means that different tokens in the same sequence are routed to experts of different sizes. 
        This is to be compared to the same metric across all tokens in the batch
        We normalise by the mean since  std(aX) = a std(X) and otherwise the metric would be biased towards sequences with a higher expert usage cost.
        This is a similar metric to the mean entropy per sequence but it takes into account the full pathway across all layers

        Args:
            router_logits (Tensor): Logits from the gate, should be a tuple of n_layers tensors of shape [batch_size*sequence_length, num_experts].
            attention_mask (Tensor): The additive attention mask for the input sequences (0 for tokens to attend to, -inf for padding tokens): shape [batch_size, sequence_length]

        Returns: 
        """
    # Correct stacking of tuple of raw_routing_weights            
        concatenated_router_logits = torch.stack(list(router_logits), dim=0) #shape (n_layers, batch * sequence_length, n_experts)
        raw_routing_weights = torch.softmax(concatenated_router_logits, dim=-1)

        n_layers,_,n_experts = raw_routing_weights.shape
        batch_size = cfg.dataloader.train.batch_size
        seq_length = cfg.tokenizer.max_length

        if len(cfg.model.expert_sizes.split(',')) > 1:

            # Compute the expert costs based on their sizes
            expert_dims = [int(size) for size in cfg.model.expert_sizes.split(',')]
            expert_dims_tensor = torch.tensor(expert_dims,dtype=torch.float32,
                device=raw_routing_weights.device )
            normalisation_factor = torch.sum(expert_dims_tensor)
            normalised_expert_sizes = expert_dims_tensor / normalisation_factor#prevent overflow
            normalised_routing_costs = normalised_expert_sizes**cfg.model.expert_cost_exponent
            normalised_routing_costs = normalised_routing_costs.to(dtype = raw_routing_weights.dtype)#cast back to original dtype

            #reshape  raw_routing_weights to (n_layers,batch_size,seq_length,n_experts) to then apply attention mask
            raw_routing_weights = raw_routing_weights.reshape(n_layers, batch_size, seq_length, n_experts)
            number_of_tokens_per_seq = seq_length
            

            #ignore padding_tokens
            if attention_mask!= None: #attention_mask has shape (Batch,seq_length) and routing_weights has shape (n_layers, batch * sequence_length, n_experts)
                multiplicative_mask = torch.where(attention_mask == 0, 1.0, 0.0).to(dtype = raw_routing_weights.dtype).reshape(1, batch_size, seq_length, 1)
                number_of_tokens_per_seq = multiplicative_mask.sum(dim=2).squeeze()
                raw_routing_weights = raw_routing_weights * multiplicative_mask
            # Compute usage loss per token
            normalised_expert_usage_loss = torch.einsum('mijk,k->mij', raw_routing_weights, normalised_routing_costs) #(n_layers,batch_size,seq_length)
            
            #sum across layers
            normalised_expert_usage_loss = torch.sum(normalised_expert_usage_loss, dim=0) #shape (batch_size,seq_length)

            # Compute RSD across tokens in the same sequence
            mean_normalised_expert_usage_loss_per_sequence = torch.sum(normalised_expert_usage_loss, dim = 1) / number_of_tokens_per_seq #shape (batch_size,)
            mean_squared_normalised_expert_usage_loss_per_sequence = torch.sum(normalised_expert_usage_loss**2, dim = 1) / number_of_tokens_per_seq #shape (batch_size,)
            intra_sequence_normalised_expert_usage_loss_var = mean_squared_normalised_expert_usage_loss_per_sequence - mean_normalised_expert_usage_loss_per_sequence**2 #shape (batch_size,)
            intra_sequence_normalised_expert_usage_loss_rel_std = torch.sqrt(intra_sequence_normalised_expert_usage_loss_var) / (mean_normalised_expert_usage_loss_per_sequence + 1e-10) #shape (batch_size,): relative std = std/mean

            # Average over sequences
            mean_intra_sequence_normalised_expert_usage_loss_rel_std = torch.mean(intra_sequence_normalised_expert_usage_loss_rel_std)


        else:
            mean_intra_sequence_normalised_expert_usage_loss_rel_std = torch.tensor(0.0, device=raw_routing_weights.device)


        return mean_intra_sequence_normalised_expert_usage_loss_rel_std #shape (1,)




    def get_normalised_expert_usage_cost_per_sequence_and_token(self,router_logits: tuple[torch.Tensor], attention_mask: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
        """_summary_
        Compute the expert usage cost per sequence in a batch. 
        Result is normalised by the  size of the experts as we do not care about the absolute value.

        Args:
            router_logits (_type_): _description_
            attention_mask (_type_): _description_
            cfg (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Correct stacking of tuple of raw_routing_weights            
        router_logits = torch.stack(list(router_logits), dim=0) #shape (n_layers, batch * sequence_length, n_experts)
        raw_routing_weights = torch.softmax(router_logits, dim=-1)

        if len(cfg.model.expert_sizes.split(',')) > 1:
            print("multiple experts")
            print(cfg.model.expert_sizes.split(','))
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
            expert_usage_loss_per_token_and_layer = torch.einsum('mijk,k->mij', raw_routing_weights, routing_costs) #(n_layers,batch_size,seq_length)
            
            #sum across layers
            expert_usage_loss_per_token = torch.sum(expert_usage_loss_per_token_and_layer, dim=0) #shape (batch_size,seq_length)
            # Average over all tokens in the sequence
            normalised_mean_expert_usage_loss_per_sequence = torch.sum(expert_usage_loss, dim = 1) / number_of_tokens_per_seq #shape (batch_size,)
        else:
            normalised_mean_expert_usage_loss_per_sequence = None

        return normalised_mean_expert_usage_loss_per_sequence #shape (batch_size,) or None if single expert

    def get_mse_per_sequence(self,logits, cfg, batch):
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
