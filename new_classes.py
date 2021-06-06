# install transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CTRLPreTrainedModel, CTRLConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast


""" PyTorch EvolvedCTRL model."""

def angle_defn(pos, i, d_model_size):
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model_size)
    return pos * angle_rates


def positional_encoding(position, d_model_size, dtype):
    # create the sinusoidal pattern for the positional encoding
    angle_rads = angle_defn(
        torch.arange(position, dtype=dtype).unsqueeze(1),
        torch.arange(d_model_size, dtype=dtype).unsqueeze(0),
        d_model_size,
    )

    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])

    pos_encoding = torch.cat([sines, cosines], dim=-1)
    return pos_encoding

class GatedConvolution(torch.nn.Module):
    def __init__(self,d_model,patch_size=3,padding=1):
        super(GatedConvolution,self).__init__()
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=2 * d_model,kernel_size=patch_size,padding=padding,bias=True)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=1)

    def forward(self,x):
        convoluted = self.conv(x.transpose(1,2)).transpose(1,2)
        out, gate = convoluted.split(int(convoluted.size(-1) / 2), -1)
        out = out * torch.sigmoid(gate)
        return out

class GLU(torch.nn.Module):
    def __init__(self,d_model,num_layers,patch_size=3,padding=1):#Dauphin's m_input= n_input= d_model
        super(GLU,self).__init__()
        self.gated_convs = nn.ModuleList([GatedConvolution(d_model,patch_size,padding) for _ in range(num_layers)])
    
    def forward(self,x):
        for convolution in self.gated_convs:
            x = convolution(x)
        return x

# evolved transformers encoder
class EvolvedTransformerEncoder(torch.nn.Module):
  def __init__(self, d_model, num_heads=8, ff_hidden=4):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(EvolvedTransformerEncoder, self).__init__()

        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=0.1) 
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
        self.dropouts = nn.ModuleList([nn.Dropout(p=0.2) for _ in range(6)])
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model,ff_hidden*d_model),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(ff_hidden*d_model,d_model)
        )
        self.glu = GLU(d_model,1)
        self.left_net = nn.Sequential(
            nn.Linear(d_model,ff_hidden*d_model),
            nn.ReLU()
        )
        self.right_net = nn.Sequential(
            nn.Conv1d(in_channels=d_model,out_channels=d_model//2,kernel_size=3,padding=1),
            nn.ReLU()
        )

        self.mid_layer_norm=nn.LayerNorm(d_model*ff_hidden)
        self.sep_conv=nn.Sequential(
            nn.Conv1d(in_channels=d_model*ff_hidden,out_channels=1,kernel_size=9,padding=4),
            nn.Conv1d(in_channels=1,out_channels=d_model,kernel_size=1)
        )


  def forward(
        self, input, mask, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False
    ):

        glued = self.glu(self.layer_norms[0](input))
        glued = self.dropouts[0](glued)
        
        glued = glued + input
        
        glu_normed = self.layer_norms[1](glued)

        left_branch = self.left_net(glu_normed)
        left_branch = self.dropouts[1](left_branch)
        
        right_branch = self.right_net(glu_normed.transpose(1,2)).transpose(1,2)
        right_branch = self.dropouts[2](right_branch)
        right_branch = F.pad(input=right_branch, pad=(0,left_branch.shape[2]-right_branch.shape[2],0,0,0,0), mode='constant', value=0)

        mid_result = left_branch+right_branch

        mid_result = self.mid_layer_norm(mid_result)
        mid_result = self.sep_conv(mid_result.transpose(1,2)).transpose(1,2)
        mid_result = self.dropouts[3](mid_result)

        mid_result = mid_result + glued

        normed = self.layer_norms[2](mid_result)
        normed=normed.transpose(0,1)
        attended = self.attention(normed,normed,normed,need_weights=False)[0].transpose(0,1)
        attended = self.dropouts[4](attended)

        attended = attended + mid_result

        normed = self.layer_norms[3](attended)
        forwarded = self.feed_forward(normed)
        forwarded = self.dropouts[5](forwarded)
        
        forwarded = forwarded + attended

        return forwarded, attended


class CTRLEvolvedModel(CTRLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.d_model_size = config.n_embd
        self.num_layers = config.n_layer/2 # evolved encoder is made of 2 layers

        self.pos_encoding = positional_encoding(config.n_positions, self.d_model_size, torch.float)

        self.w = nn.Embedding(config.vocab_size, config.n_embd)

        self.dropout = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [EvolvedTransformerEncoder(config.n_embd, config.n_head, 2) for _ in range(config.n_layer)]
            # [EncoderLayer(config.n_embd, config.n_head, config.dff, config.resid_pdrop) for _ in range(config.n_layer)]
        )
        self.layernorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.w

    def set_input_embeddings(self, new_embeddings):
        self.w = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].multi_head_attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
            token_type_embeds = self.w(token_type_ids)
            token_type_embeds *= np.sqrt(self.d_model_size)
        else:
            token_type_embeds = 0
        position_ids = position_ids.view(-1, input_shape[-1])

        if inputs_embeds is None:
            inputs_embeds = self.w(input_ids)
        # inputs_embeds = embedded.unsqueeze(0) if len(input_ids.shape)<2 else embedded
        seq_len = input_shape[-1]
        mask = torch.triu(torch.ones(seq_len + past_length, seq_len + past_length), 1).to(device)

        inputs_embeds *= np.sqrt(self.d_model_size)

        pos_embeds = self.pos_encoding[position_ids, :].to(device)

        hidden_states = inputs_embeds + pos_embeds + token_type_embeds

        hidden_states = self.dropout(hidden_states)

        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, (h, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = h(
                hidden_states,
                mask,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states, present = outputs
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions += (outputs[2],)

        hidden_states = self.layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class CTRLLMHeadEvolvedModel(CTRLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = CTRLEvolvedModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, use_cache=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {"input_ids": input_ids, "past_key_values": past, "use_cache": use_cache}

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
       
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )