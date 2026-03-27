import torch
import torch.nn as nn
from transformers import (
    AutoFeatureExtractor, 
    WavLMModel, 
    HubertModel, 
    Wav2Vec2Model)
from utils import custom_print


class BackboneSFM(nn.Module):
    """
    Backbone Speech Foundation Model (SFM) Class
    """
    def __init__(self, params):
        """
        Args:
            params
        """
        super().__init__()
        self.params = params
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.params.feature_extractor_name)
        if self.params.model_name.split('/')[-1] == "wavlm-base":
            self.backbone = WavLMModel.from_pretrained(self.params.model_name)
        elif self.params.model_name.split('/')[-1] == "hubert-base":
            self.backbone = HubertModel.from_pretrained(self.params.model_name)
        elif self.params.model_name.split('/')[-1] == "wav2vec2-base":
            self.backbone = Wav2Vec2Model.from_pretrained(self.params.model_name)
        else:
            raise Exception("\nBackbone model class definition error.")
        self.config = self.backbone.config

    def get_hidden_state(self, x, attention_mask=None, total=False):
        """
        Args:
            x
            attention_mask
            total
        Returns:
            tensor
        """
        outputs = self.backbone(
            x.to(self.params.device), 
            attention_mask=attention_mask, 
            output_hidden_states=total)
        if total:
            # Dimensions: [num_layers, batch, time, emb_size]
            return torch.stack(outputs.hidden_states)
        return outputs.last_hidden_state


class ProjectorLinear(BackboneSFM):
    """
    Linear SFM Projectior Class
    """
    def __init__(self, params):

        super().__init__(params)
        hidden_size = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=params.projector_out_dim),
            nn.LayerNorm(normalized_shape=params.projector_out_dim))

    def forward(self, x):
        """
        Args:
            x
        Returns:
            tensor
        """
        x_proj = self.proj(x)
        return nn.functional.normalize(x_proj, p=2, dim=1)


class ProjectorNonLinear(BackboneSFM):
    """
    Nonlinear SFM Projectior Class 
    """
    def __init__(self, params):

        super().__init__(params)
        self.input_size = self.backbone.config.hidden_size
        if params.stat_pooling:
            self.input_size *= 2
        self.proj_size = int(self.input_size//3)
        self.proj = nn.Sequential(
            nn.Linear(self.input_size, self.proj_size),
            nn.LayerNorm(self.proj_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.proj_size, params.projector_out_dim),
            nn.LayerNorm(params.projector_out_dim)
        )
        self.residual = nn.Linear(self.input_size, params.projector_out_dim) if self.input_size != params.projector_out_dim else nn.Identity()
        self.norm = nn.LayerNorm(params.projector_out_dim)

    def forward(self, x):
        """
        Args:
            x
        Returns:
            tensor
        """
        x_resid = self.residual(x)
        x_proj = self.proj(x)
        x_proj += x_resid
        x_norm = self.norm(x_proj)
        return nn.functional.normalize(x_norm, p=2, dim=1)


class AdapterLinear(BackboneSFM):
    """
    Linear SFM Adapter Class
    """
    def __init__(self, params):

        super().__init__(params)
        hidden_size = self.backbone.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2)
            # nn.LayerNorm(hidden_size // 2) # nn.BatchNorm1d(hidden_size // 2)
        )
        self.classifier = nn.Linear(hidden_size, self.params.num_classes) # hidden_size // 2
        self.dropout = nn.Dropout(0.2)

    def get_normalized_projections(self, x):
        """
        Args:
            x
        Returns:
            tensor
        """
        embs_proj = self.projection(x)
        return nn.functional.normalize(embs_proj, p=2, dim=1)

    def forward(self, x, attention_mask=None, return_embeddings=False):
        """
        Args:
            x
            attention_mask
            return_embeddings
        Returns:
            if return_embeddings:
                logits
                preds
                embs_norm
            else:
                logits, preds
        """
        hs = self.get_hidden_state(x, attention_mask)
        if attention_mask is None:
            pooled_output = torch.mean(hs, dim=1)
        else:
            input_lengths = attention_mask.sum(dim=-1)
            pooled_output = torch.sum(hs * attention_mask.unsqueeze(-1), dim=1)
            pooled_output = pooled_output / input_lengths.unsqueeze(-1)
        logits = self.classifier(self.dropout(pooled_output))
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        if return_embeddings:
            embs_norm = self.get_normalized_projections(pooled_output)
            return logits, preds, embs_norm
        return logits, preds


class AdapterMLP(BackboneSFM):
    """
    Nonlinear MLP-based SFM Adapter Class
    """
    def __init__(self, params):

        super().__init__(params)
        hidden_size = self.sfm.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2), # 768 --> 384
            # nn.BatchNorm1d(hidden_size//2), # ?
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_size//2, self.params.num_classes) # 384 --> N
        )

    def forward(self, x, attention_mask=None):
        """
        Args:
            x
            attention_mask
        Returns:
            logits
            preds
        """
        hs = self.get_hidden_state(x)
        if attention_mask is None:
            pooled_output = torch.mean(hs, dim=1)
        else:
            input_lengths = attention_mask.sum(dim=-1)
            pooled_output = torch.sum(hs * attention_mask.unsqueeze(-1), dim=1)
            pooled_output = pooled_output / input_lengths.unsqueeze(-1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return logits, preds


class AdapterHybrid(BackboneSFM):
    """
    Hybrid SFM Adapter Class
    """
    def __init__(self, params):
        
        super().__init__(params)
        self.params.projector_out_dim = 128
        if 0: # use linear
            self.projector = ProjectorLinear(self.params)
        else:
            self.projector = ProjectorNonLinear(self.params)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.params.projector_out_dim, self.params.num_classes))

    def forward(self, x, attention_mask=None, return_embeddings=False, stat_pooling=True):
        """
        Args:
            ...
        Returns:
            ...
        """
        hs = self.get_hidden_state(x, attention_mask)
        if stat_pooling:
            pooled_embs_mean = torch.mean(hs, dim=1)
            pooled_embs_std = torch.std(hs, dim=1)
            pooled_embs = torch.cat([pooled_embs_mean, pooled_embs_std], dim=1)
        else:
            pooled_embs = torch.mean(hs, dim=1)
        embs_norm = self.projector(pooled_embs)
        logits = self.classifier(embs_norm)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        if return_embeddings:
            return logits, preds, embs_norm
        return logits, preds


class ProbingAdapterHybrid(BackboneSFM):
    """
    Probing Hybrid SFM Adapter Class
    """
    def __init__(self, params):

        super().__init__(params)
        for p in self.sfm.parameters():
            p.requires_grad = False
        self.fusion_method = params.fusion_method
        self.sfm_num_layers = self.sfm.config.num_hidden_layers + 1
        self.sfm_hidden_size = self.sfm.config.hidden_size
        if self.fusion_method is not None:
            self.topk = params.topk_layers
            num_weights = len(self.topk) if self.topk is not None else self.sfm_num_layers
            if self.fusion_method == "gff": # Gated Feature Fusion
                custom_print("\nInit Gated Feature Fusion...")
                self.feature_fusion = nn.Linear(self.sfm_hidden_size, num_weights) # gff_gate
                nn.init.zeros_(self.feature_fusion.weight)
                nn.init.zeros_(self.feature_fusion.bias)
            elif self.fusion_method == "tws": # Trainable weighted sum
                custom_print("\nInit Trainable Weighted Sum...")
                self.feature_fusion = nn.Parameter(torch.ones(num_weights))
        if self.params.projector_out_dim is not None:
            custom_print("\nInit Projector...")
            self.projector = ProjectorNonLinear(self.params)
            clf_input_dim = self.params.projector_out_dim
        else:
            clf_input_dim = self.sfm_hidden_size
            if params.stat_pooling: clf_input_dim *= 2
        custom_print("\nInit Linear classifier...")
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(clf_input_dim, self.params.num_classes)
            )

    def forward(self, x, attention_mask=None, return_embeddings=False, stat_pooling=True):
        """
        L:Layer, B:Batch, T:Time, D:Dimension
        """
        if self.fusion_method is not None:
            hidden_states = self.get_hidden_state(x, attention_mask, total=True) # [L, B, T, D]
            if self.topk is not None: hidden_states = hidden_states[self.topk]
            if self.fusion_method == "gff": # Gated Feature Fusion
                global_context = hidden_states.mean(dim=0).mean(dim=1) # [B, D]
                weights = torch.softmax(self.feature_fusion(global_context), dim=-1) # [B, L]
                # Applying weights: [B, L, 1, 1] * [B, L, T, D] -> [B, T, D]
                features = (hidden_states.transpose(0, 1) * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            elif self.fusion_method == "tws": # Trainable weighted sum
                weights = torch.softmax(self.feature_fusion, dim=0) # [L]
                features = (hidden_states * weights.view(-1, 1, 1, 1)).sum(dim=0) # [B, T, D]
            elif self.fusion_method == "mean": # Mean fusion
                features = hidden_states.mean(dim=0) # [B, T, D]
            else:
                raise Exception("Fusion method error during forward!")
        else: # Last layer
            features = self.get_hidden_state(x, attention_mask, total=False) # [B, T, D]

        if self.params.stat_pooling:
            embs = torch.cat([
                torch.mean(features, dim=1),
                torch.std(features, dim=1)], dim=-1) # [B, 2D]
        else:
            embs = torch.mean(features, dim=1) # [B, D]

        if self.params.projector_out_dim is not None: embs = self.projector(embs)

        logits = self.classifier(embs)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        if return_embeddings:
            return logits, preds, embs
        return logits, preds


def make_model(args):
    """
    Args:
        ...
    Returns:
        ...
    """
    custom_print("\nModel initialization...")
    if args.topk_layers: # TODO
        if args.dataset_name == "iemocap":
            args.topk_layers = [0, 6, 7, 9, 10, 11]
        elif args.dataset_name == "emotiontalk":
            args.topk_layers = [0, 5, 6, 7, 12]
        else:
            raise Exception(f"Config conflict: top k layers aggregation indexces for {args.dataset_name} is unknown.")
        custom_print(f"\nIndexes of top k encoder layers aggregation from linear probing: {args.topk_layers}")
    else:
        args.topk_layers = None
        custom_print(f"\nUse all encoder layers to compute the output embedding.")

    adapter_map = {
        "linear": AdapterLinear,
        "mlp": AdapterMLP,
        "hybrid": AdapterHybrid,
        "hybrid_probing": ProbingAdapterHybrid # projector + classifier
    }
    adapter_type = adapter_map.get(args.adapter)
    
    if adapter_type:
        model = adapter_type(args)
    else:
        raise ValueError(f"Unknown adapter type:{args.adapter}")
    custom_print(f"Initialized adapter: {args.adapter}")
    return model.to(model.params.device)
