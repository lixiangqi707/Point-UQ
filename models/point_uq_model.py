from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import uni3D.models.uni3d as uni3d_models


def _build_uni3d_args(args):
    return SimpleNamespace(
        model="create_uni3d",
        pc_model="eva02_base_patch14_448",
        pretrained_pc="",
        drop_path_rate=0.0,
        pc_feat_dim=768,
        embed_dim=1024,
        group_size=64,
        num_group=512,
        pc_encoder_dim=512,
        patch_dropout=0.0,
        ckpt_path=args.uni3d_ckpt_path,
    )


class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=1.0):
        super().__init__()
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn(original_layer.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, original_layer.out_features))
        self.scaling = alpha / rank

        self.register_parameter("weight", self.original_layer.weight)
        if hasattr(self.original_layer, "bias"):
            self.register_parameter("bias", self.original_layer.bias)
        else:
            self.bias = None

    def forward(self, x):
        orig_output = self.original_layer(x)
        lora_output = (x @ self.lora_A) @ self.lora_B * self.scaling
        return orig_output + lora_output


class FeatureEnhancementModule(nn.Module):
    def __init__(
        self,
        feature_dim=1024,
        num_heads=8,
        hidden_dim=512,
        num_layers=2,
        lora_rank=8,
        lora_alpha=1.0,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.residual_proj = LoRALayer(nn.Linear(feature_dim, feature_dim), rank=lora_rank, alpha=lora_alpha)

    def forward(self, pc_features, all_features):
        layers = list(range(12))
        mid_features = [all_features[f"layer{layer}"] for layer in layers]
        stacked_features = torch.stack(mid_features, dim=1)
        query_features = pc_features.unsqueeze(1)
        combined_features = torch.cat([query_features, stacked_features], dim=1)
        fused_features = self.transformer_encoder(combined_features)
        enhanced_feature = fused_features[:, 0, :]
        return pc_features + self.residual_proj(enhanced_feature)


class COS_Model(nn.Module):
    def __init__(
        self,
        args,
        device=None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.cache_features = {}
        self.lambda_opt = 2.0
        self.beta = 5.0

        uni3d_args = _build_uni3d_args(args)
        self.uni3d_model = getattr(uni3d_models, uni3d_args.model)(args=uni3d_args).to(self.device)
        checkpoint = torch.load(uni3d_args.ckpt_path, map_location="cpu")
        self.uni3d_model.load_state_dict(checkpoint["module"])
        for param in self.uni3d_model.parameters():
            param.requires_grad = False

        self.featureEnhancementModule = FeatureEnhancementModule().to(self.device)

    def update_cache(self, class_idx, cache_feature):
        if isinstance(cache_feature, np.ndarray):
            cache_feature = torch.tensor(cache_feature, dtype=torch.float32, device=self.device)
        elif isinstance(cache_feature, torch.Tensor):
            cache_feature = cache_feature.detach().to(self.device, dtype=torch.float32)
        else:
            raise TypeError("cache_feature must be a numpy array or torch tensor")

        cache_key = str(class_idx)
        if cache_key not in self.cache_features:
            self.cache_features[cache_key] = nn.Parameter(cache_feature, requires_grad=False)
        else:
            self.cache_features[cache_key].data.copy_(cache_feature)

    def update_classifier_cache(self, avg_features):
        for class_idx, avg_feature in avg_features.items():
            self.update_cache(class_idx, avg_feature)

    def _class_centers(self):
        if not self.cache_features:
            raise RuntimeError("Classifier cache is empty. Call update_classifier_cache first.")
        ordered = [self.cache_features[key] for key in sorted(self.cache_features.keys(), key=lambda item: int(item))]
        return torch.stack(ordered)

    def entropy(self, features):
        prob = F.softmax(features, dim=-1)
        return -torch.sum(prob * torch.log(prob + 1e-8), dim=-1)

    def uncertainty_activation(self, entropy):
        return torch.sigmoid(self.lambda_opt * entropy.clamp(min=0))

    def extract_point_features(self, points, rgb):
        feature = torch.cat((points, rgb), dim=-1)
        pc_features, all_features = self.uni3d_model.encode_pc_all(
            feature,
            return_all=True,
            output_blocks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        )
        pc_features = F.normalize(pc_features, p=2, dim=-1)
        pc_features_enhanced = self.featureEnhancementModule(pc_features, all_features)
        return pc_features, pc_features_enhanced

    def forward(
        self,
        points,
        rgb,
        prompts_feats,
        task_id=0,
        label=None,
    ):
        pc_features, pc_features_enhanced = self.extract_point_features(points, rgb)
        class_centers = self._class_centers()
        logits = pc_features_enhanced.float() @ prompts_feats.float().t()
        cosine_sim_matrix = F.cosine_similarity(
            pc_features.unsqueeze(1),
            class_centers.unsqueeze(0),
            dim=-1,
        )

        if label is None:
            zero = torch.zeros((), device=pc_features.device)
            if task_id == 0:
                return logits, zero
            feature_entropy = self.entropy(logits)
            alpha_final = self.uncertainty_activation(feature_entropy).unsqueeze(-1)
            final_logits = alpha_final * cosine_sim_matrix + (1 - alpha_final) * logits
            return final_logits, zero

        loss_cosine = F.cross_entropy(cosine_sim_matrix, label)
        loss_logits = F.cross_entropy(logits, label)
        total_loss = loss_cosine + self.beta * loss_logits

        if task_id == 0:
            return logits, total_loss

        feature_entropy = self.entropy(logits)
        alpha_final = self.uncertainty_activation(feature_entropy).unsqueeze(-1)
        final_logits = alpha_final * cosine_sim_matrix + (1 - alpha_final) * logits

        return final_logits, total_loss
