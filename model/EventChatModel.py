import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as init
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import LlamaForCausalLM, LlamaConfig, LlamaModel
from transformers import LlamaTokenizer, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType 
from dataset.constants import IGNORE_INDEX, EVENT_TOKEN_INDEX, DEFAULT_EVENT_TOKEN, DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN, DEFAULT_EVENT_PATCH_TOKEN
from typing import List, Optional, Tuple, Union, Dict, Callable
from transformers.generation.utils import GenerateOutput
import numpy as np
import torch

def get_spatio_temporal_features(features, num_temporal_tokens=None):
    if isinstance(features, list):
        features = torch.stack(features)

    if features.ndim != 3:
        raise ValueError("Input features should be a 3D tensor with shape (t, s, c)")

    t, _, _ = features.shape

    if num_temporal_tokens is None:
        num_temporal_tokens = t

    temporal_tokens = features.mean(dim=1)

    if num_temporal_tokens > t:
        padding_size = num_temporal_tokens - t
        temporal_tokens = torch.nn.functional.pad(temporal_tokens, (0, 0, 0, padding_size))
    elif num_temporal_tokens < t:
        temporal_tokens = temporal_tokens[:num_temporal_tokens]

    spatial_tokens = features.mean(dim=0)
    sp_features = torch.cat([temporal_tokens, spatial_tokens], dim=0)

    return sp_features.to(features.device)  


class EventChatConfig(LlamaConfig):
    model_type = "EventChat_llama" 


class VisualTower(nn.Module):
    def __init__(self, visual_tower):
        super().__init__()

        self.visual_tower_name = visual_tower
        self.event_processor = CLIPImageProcessor.from_pretrained(self.visual_tower_name)
        self.visual_tower = CLIPVisionModel.from_pretrained(self.visual_tower_name, torch_dtype=torch.bfloat16)
        self.visual_tower.requires_grad_(False)
    
    def forward(self, event_tensor):
        outputs = self.visual_tower.vision_model(event_tensor)
        events_feature = outputs.last_hidden_state
        events_feature = self.visual_projecotor(events_feature)

        return events_feature

class EventChatLlamaModel(LlamaModel):
    config_class = EventChatConfig

    def __init__(self, config: LlamaConfig):
        super(EventChatLlamaModel, self).__init__(config)

        self.mlp_depth = 2
        self.text_hidden_size = 1024
        self.hidden_size = 4096

        if hasattr(config, "mm_visual_tower"):          
            self.visual_tower = self.build_visual_tower(config.mm_visual_tower)
            self.visual_projector = self.build_mlp_projector(self.text_hidden_size, self.hidden_size).to(dtype=torch.bfloat16)

        if hasattr(config, "event_feature_adaptor"):
            self.feature_adaptor = nn.Linear(self.hidden_size, self.hidden_size)

        if hasattr(config, "use_event_qformer"):
            self.query_embeddings, self.attention_layers = self.build_event_qformer(config)
            self.register_parameter('query_embeddings', self.query_embeddings)  
            self.attention_layers = nn.ModuleList(self.attention_layers)

    def build_visual_tower(self, visual_tower):
        return VisualTower(visual_tower)
        

    def build_mlp_projector(self, text_hidden_size, hidden_dim):
        mlp_depth = self.mlp_depth
        modules = [nn.Linear(text_hidden_size, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*modules)
    
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def initialize_vision_modules(self, model_args, fsdp=None):
        visual_tower = model_args.vision_tower
        self.config.mm_visual_tower = visual_tower
        self.config.event_feature_adaptor = True

        # Build the visual tower
        visual_tower = self.build_visual_tower(model_args.vision_tower) 
        self.visual_tower = visual_tower

        # Build the visual projector
        self.visual_projector = self.build_mlp_projector(self.text_hidden_size, self.hidden_size).to(dtype=torch.bfloat16)

        # Load feature adaptor if needed
        if model_args.use_feature_adaptor:
            self.feature_adaptor = nn.Linear(self.hidden_size, self.hidden_size)

        # Load event Qformer if needed
        if model_args.use_event_qformer:
            self.query_embedder, self.attention_layers = self.build_event_qformer(model_args)
            self.add_module("query_embedder", self.query_embedder)
            self.attention_layers = nn.ModuleList(self.attention_layers)
        
        # Load pretrained weights for feature_adaptor if provided
        if model_args.pretrain_feature_adaptor is not None:
            print("Loading feature_adaptor pretrain weights...")
            pretrained_weights = torch.load(model_args.pretrain_feature_adaptor)
            # Adjust keys to match model structure
            pretrained_weights = {k.replace("model.feature_adaptor.", ""): v for k, v in pretrained_weights.items()}
            self.feature_adaptor.load_state_dict(pretrained_weights, strict=True)
            print("Pretrained weights loaded successfully into feature_adaptor.")

        # Load pretrained weights for visual_projector if provided
        if model_args.pretrain_mm_mlp_adapter is not None:
            print("Loading mm_projector pretrain weights...")
            pretrained_weights = torch.load(model_args.pretrain_mm_mlp_adapter)
            # Adjust keys to match model structure
            pretrained_weights = {k.replace("model.visual_projector.", ""): v for k, v in pretrained_weights.items()}
            self.visual_projector.load_state_dict(pretrained_weights, strict=True)
            print("Pretrained weights loaded successfully into visual_projector.")

        # Load pretrained weights for query_embedder if specified
        if model_args.pretrain_query_embedder is not None:
            print("Loading query_embedder pretrain weights...")
            pretrained_weights = torch.load(model_args.pretrain_query_embedder)
            pretrained_weights = {k.replace("model.query_embedder.", ""): v for k, v in pretrained_weights.items()}
            # Load query_embedder weights
            self.query_embedder.load_state_dict(pretrained_weights, strict=True)
            print("Pretrained weights loaded successfully into query_embedder.")

        # Load pretrained weights for attention_layers if specified
        if model_args.pretrain_attention_layers is not None:
            print("Loading attention_layers pretrain weights...")
            pretrained_weights = torch.load(model_args.pretrain_attention_layers)
            
            # Filter the pretrained weights to only include attention layers' weights
            attention_layer_weights = {k: v for k, v in pretrained_weights.items() if "attention_layers" in k}
            
            # Ensure we are only loading weights for the attention layers
            for i, attention_layer in enumerate(self.attention_layers):
                # Match keys for each attention layer and load the state dict
                layer_weights = {k.replace(f"model.attention_layers.{i}.", ""): v for k, v in attention_layer_weights.items() if f"attention_layers.{i}" in k}
                attention_layer.load_state_dict(layer_weights, strict=True)
            print("Pretrained weights loaded successfully into attention_layers.")


class EventChatModel(LlamaForCausalLM):

    config_class = EventChatConfig

    def __init__(self, config) -> None:
        super(LlamaForCausalLM, self).__init__(config)
        
        self.model = EventChatLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model
    
    def get_visual_tower(self):
        return self.get_model().visual_tower
    
    
    def visval_encode(self, event_tensor):
        with torch.no_grad():
            outputs = self.get_model().visual_tower.visual_tower(event_tensor)
        events_feature = outputs.last_hidden_state
        events_feature = events_feature.detach().requires_grad_(True)
        events_feature = self.get_model().visual_projector(events_feature)
        return events_feature

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_EVENT_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        event_tensors: Optional[torch.Tensor] = None,
        event_image_sizes: Optional[torch.Tensor] = None,
        event_features: Optional[torch.Tensor] = None,  # NEW: cached features for Stage 4
        event_feature = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """Generate text tokens.

        Args:
            event_tensors: Raw event tensors (Stage 3+4: vision encoding + decoding)
            event_features: Pre-computed vision features (Stage 4 only: decoding with cached features)
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if event_tensors is not None or event_features is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                event_tensors=event_tensors,
                event_image_sizes=event_image_sizes,
                event_features=event_features  # Pass cached features if provided
            )
        else:
            raise NotImplementedError("please input Event")

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        event_tensors = kwargs.pop("event_tensors", None)
        event_image_sizes = kwargs.pop("event_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if event_tensors is not None:
            inputs['event_tensors'] = event_tensors
        if event_image_sizes is not None:
            inputs['event_image_sizes'] = event_image_sizes
        return inputs
        
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        event_tensors=None, event_image_sizes=None, event_features=None
    ):
        """Prepare inputs for multimodal generation.

        Args:
            event_tensors: Raw event image tensors (will be encoded in Stage 3)
            event_features: Pre-computed event features (Stage 4 only, skip Stage 3)
        """
        if event_tensors is None and event_features is None:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Stage 3: Vision encoding (skip if event_features provided)
        if event_features is None:
            # Full path: encode event tensors
            if isinstance(event_tensors, list):
                if not all(isinstance(item, list) for item in event_tensors):
                    event_tensors = [event_tensors]

                ev_features_list = []
                for item in event_tensors:
                    ev_feature = []
                    for ev in item:
                        ev = ev.unsqueeze(0)
                        feature = self.visval_encode(ev)
                        feature = self.get_model().feature_adaptor(feature)
                        feature = feature.squeeze(0)
                        ev_feature.append(feature)
                    event_feature = get_spatio_temporal_features(ev_feature)
                    ev_features_list.append(event_feature)
                event_features = torch.stack(ev_features_list)
            else:
                event_features = self.visval_encode(event_tensors)
        else:
            # Cached path: event_features already computed (Stage 4 only)
            # Apply feature adaptor if needed
            if hasattr(self.get_model(), 'feature_adaptor'):
                if isinstance(event_features, list):
                    event_features = [self.get_model().feature_adaptor(f) for f in event_features]
                else:
                    event_features = self.get_model().feature_adaptor(event_features)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_event_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_events = (cur_input_ids == EVENT_TOKEN_INDEX).sum()
            if num_events == 0:
                cur_event_features = event_features[cur_event_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_event_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_event_idx += 1
                continue
            
            event_token_indices = [-1] + torch.where(cur_input_ids == EVENT_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(event_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[event_token_indices[i]+1:event_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[event_token_indices[i]+1:event_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_events + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_events:
                    cur_event_features = event_features[cur_event_idx]
                    cur_event_idx += 1
                    cur_new_input_embeds.append(cur_event_features)
                    cur_new_labels.append(torch.full((cur_event_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = 2048
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            tokenizer_padding_side = 'right'
            if  tokenizer_padding_side == 'left':
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

      
AutoConfig.register("EventChat_llama", EventChatConfig)
AutoModelForCausalLM.register(EventChatConfig, EventChatModel)
