"""
Multi-Modal Deep Learning Model for the Avito Demand Prediction Challenge.

Architecture:
    ┌─────────────┐  ┌──────────────┐  ┌─────────────┐
    │ Tabular Tower│  │  Text Tower  │  │ Image Tower │
    │  (Embeddings │  │  (Projection │  │(Projection  │
    │  + ResidMLP) │  │  + AttnPool) │  │   + NoImg)  │
    └──────┬──────┘  └──────┬───────┘  └──────┬──────┘
           │                │                  │
           └────────┬───────┴──────────┬───────┘
                    │                  │
              ┌─────▼──────────────────▼─────┐
              │  Cross-Modal Gated Attention  │
              └──────────────┬───────────────┘
                             │
                    ┌────────▼────────┐
                    │ Residual MLP    │
                    │   Head          │
                    └────────┬────────┘
                             │
                       ┌─────▼─────┐
                       │  Sigmoid  │
                       └───────────┘
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MODEL_CFG, DATA_CFG, FEATURES_DIR


# ============================================================================
# Building Blocks
# ============================================================================

class GhostBatchNorm(nn.Module):
    """
    Ghost Batch Normalization (from TabNet paper).
    Applies BatchNorm on virtual mini-batches within each real batch.
    Provides implicit regularization for tabular data.
    """

    def __init__(self, num_features: int, ghost_batch_size: int = 64, momentum: float = 0.01):
        super().__init__()
        self.num_features = num_features
        self.ghost_batch_size = ghost_batch_size
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_size = x.size(0)
            if batch_size <= self.ghost_batch_size:
                return self.bn(x)

            # Split into ghost batches
            chunks = x.split(self.ghost_batch_size, dim=0)
            normed = [self.bn(chunk) for chunk in chunks]
            return torch.cat(normed, dim=0)
        else:
            return self.bn(x)


class ResidualBlock(nn.Module):
    """Residual MLP block with Ghost Batch Norm."""

    def __init__(self, dim: int, dropout: float = 0.3, ghost_batch_size: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            GhostBatchNorm(dim, ghost_batch_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            GhostBatchNorm(dim, ghost_batch_size),
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(x + self.block(x)))


# ============================================================================
# Tower 1: Tabular
# ============================================================================

class TabularTower(nn.Module):
    """
    Processes structured tabular features:
    - Entity embeddings for categorical features
    - Concatenation with numerical features
    - Residual MLP with Ghost Batch Normalization
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        cat_cols: List[str],
        num_numerical: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        ghost_batch_size: int = 64,
    ):
        super().__init__()

        # Create embedding layers
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0

        for col in cat_cols:
            num_cats = vocab_sizes[col]
            embed_dim = min(50, (num_cats + 1) // 2)
            embed_dim = max(embed_dim, 4)  # At least 4 dims
            self.embeddings[col] = nn.Embedding(
                num_embeddings=num_cats + 1,  # +1 for unknown
                embedding_dim=embed_dim,
                padding_idx=0,
            )
            total_embed_dim += embed_dim

        self.cat_cols = cat_cols
        input_dim = total_embed_dim + num_numerical

        # Build MLP
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                GhostBatchNorm(h_dim, ghost_batch_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.residual = ResidualBlock(hidden_dims[-1], dropout, ghost_batch_size)
        self.output_dim = hidden_dims[-1]

    def forward(self, cat_features: torch.Tensor, num_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cat_features: (batch, num_cat_cols) integer tensor
            num_features: (batch, num_numerical) float tensor
        Returns:
            (batch, output_dim) tensor
        """
        # Embed categoricals
        embeds = []
        for i, col in enumerate(self.cat_cols):
            # Clamp to valid range
            col_data = cat_features[:, i].clamp(
                0, self.embeddings[col].num_embeddings - 1
            )
            embeds.append(self.embeddings[col](col_data))

        cat_emb = torch.cat(embeds, dim=-1)

        # Concatenate with numericals
        x = torch.cat([cat_emb, num_features], dim=-1)

        # MLP + Residual
        x = self.mlp(x)
        x = self.residual(x)

        return x


# ============================================================================
# Tower 2: Text
# ============================================================================

class TextTower(nn.Module):
    """
    Processes pre-computed text embeddings from sentence-transformers.
    Applies learned projections and attention-weighted pooling over
    title and description embeddings.
    """

    def __init__(
        self,
        input_dim: int = 384,
        projection_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Separate projections for title and description
        self.title_proj = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.desc_proj = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Attention over title & description
        self.attention = nn.Sequential(
            nn.Linear(projection_dim, 1),
        )

        self.output_dim = projection_dim

    def forward(self, title_emb: torch.Tensor, desc_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            title_emb: (batch, text_embed_dim) from sentence-transformer
            desc_emb: (batch, text_embed_dim) from sentence-transformer
        Returns:
            (batch, projection_dim) tensor
        """
        title_h = self.title_proj(title_emb)  # (B, proj_dim)
        desc_h = self.desc_proj(desc_emb)      # (B, proj_dim)

        # Stack: (B, 2, proj_dim)
        stacked = torch.stack([title_h, desc_h], dim=1)

        # Attention weights: (B, 2, 1)
        attn_logits = self.attention(stacked)
        attn_weights = F.softmax(attn_logits, dim=1)

        # Weighted sum: (B, proj_dim)
        output = (stacked * attn_weights).sum(dim=1)

        return output


# ============================================================================
# Tower 3: Image
# ============================================================================

class ImageTower(nn.Module):
    """
    Processes pre-computed image features from EfficientNet-B0.
    Uses a learned "no-image" embedding for missing images.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        projection_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

        # Learned embedding for missing images
        self.no_image_embedding = nn.Parameter(
            torch.randn(projection_dim) * 0.02
        )

        self.output_dim = projection_dim

    def forward(self, img_features: torch.Tensor, img_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_features: (batch, image_feature_dim) from EfficientNet
            img_mask: (batch,) boolean - True if image exists
        Returns:
            (batch, projection_dim) tensor
        """
        projected = self.projection(img_features)  # (B, proj_dim)

        # Replace missing images with learned embedding
        no_img = self.no_image_embedding.unsqueeze(0).expand_as(projected)
        mask = img_mask.unsqueeze(-1)  # (B, 1)
        output = torch.where(mask, projected, no_img)

        return output


# ============================================================================
# Fusion Module
# ============================================================================

class GatedAttentionFusion(nn.Module):
    """
    Cross-modal gated attention fusion.
    Dynamically weights the contribution of each modality per sample.
    """

    def __init__(
        self,
        modality_dim: int = 128,
        num_modalities: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Gate network: takes concatenation of all modalities
        self.gate = nn.Sequential(
            nn.Linear(modality_dim * num_modalities, modality_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(modality_dim, num_modalities),
        )

        # Optional: transform after fusion
        self.post_fusion = nn.Sequential(
            nn.LayerNorm(modality_dim),
            nn.Dropout(dropout),
        )

        self.output_dim = modality_dim

    def forward(self, *modality_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *modality_outputs: Variable number of (batch, modality_dim) tensors
        Returns:
            (batch, modality_dim) fused tensor
        """
        # Concatenate all modalities: (B, num_modalities * dim)
        concat = torch.cat(modality_outputs, dim=-1)

        # Compute gating weights: (B, num_modalities)
        gate_weights = F.softmax(self.gate(concat), dim=-1)

        # Stack modalities: (B, num_modalities, dim)
        stacked = torch.stack(modality_outputs, dim=1)

        # Weighted sum: (B, dim)
        fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)

        return self.post_fusion(fused)


# ============================================================================
# Full Model
# ============================================================================

class AvitoMultiModalModel(nn.Module):
    """
    Complete multi-modal model for Avito Demand Prediction.
    Combines tabular, text, and image towers with gated attention fusion.
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        cat_cols: List[str],
        num_numerical: int,
        text_embed_dim: int = 384,
        image_feature_dim: int = 1280,
        tabular_hidden_dims: List[int] = [256, 128],
        text_projection_dim: int = 128,
        image_projection_dim: int = 128,
        head_hidden_dims: List[int] = [64, 32],
        dropout: float = 0.3,
        ghost_batch_size: int = 64,
    ):
        super().__init__()

        # --- Towers ---
        self.tabular_tower = TabularTower(
            vocab_sizes=vocab_sizes,
            cat_cols=cat_cols,
            num_numerical=num_numerical,
            hidden_dims=tabular_hidden_dims,
            dropout=dropout,
            ghost_batch_size=ghost_batch_size,
        )

        self.text_tower = TextTower(
            input_dim=text_embed_dim,
            projection_dim=text_projection_dim,
            dropout=dropout,
        )

        self.image_tower = ImageTower(
            input_dim=image_feature_dim,
            projection_dim=image_projection_dim,
            dropout=dropout,
        )

        # Ensure all towers output same dimension for fusion
        assert self.tabular_tower.output_dim == text_projection_dim == image_projection_dim, \
            "All tower output dimensions must match for gated attention fusion"

        fusion_dim = text_projection_dim

        # --- Fusion ---
        self.fusion = GatedAttentionFusion(
            modality_dim=fusion_dim,
            num_modalities=3,
            dropout=dropout,
        )

        # --- Prediction Head ---
        head_layers = []
        prev_dim = fusion_dim
        for h_dim in head_hidden_dims:
            head_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        head_layers.append(nn.Linear(prev_dim, 1))
        self.prediction_head = nn.Sequential(*head_layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Kaiming initialization for linear layers, normal for embeddings."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: Dictionary containing:
                - cat_features: (B, num_cat) int tensor
                - num_features: (B, num_numerical) float tensor
                - title_emb: (B, text_embed_dim) float tensor
                - desc_emb: (B, text_embed_dim) float tensor
                - img_features: (B, image_feature_dim) float tensor
                - img_mask: (B,) bool tensor
        Returns:
            (B,) predicted deal_probability ∈ [0, 1]
        """
        # Tower outputs
        h_tabular = self.tabular_tower(batch["cat_features"], batch["num_features"])
        h_text = self.text_tower(batch["title_emb"], batch["desc_emb"])
        h_image = self.image_tower(batch["img_features"], batch["img_mask"])

        # Fusion
        h_fused = self.fusion(h_tabular, h_text, h_image)

        # Prediction
        logits = self.prediction_head(h_fused).squeeze(-1)
        predictions = torch.sigmoid(logits)

        return predictions

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model() -> AvitoMultiModalModel:
    """
    Build the model using saved vocab sizes and config.
    """
    # Load vocab sizes
    with open(FEATURES_DIR / "vocab_sizes.json") as f:
        vocab_sizes = json.load(f)

    model = AvitoMultiModalModel(
        vocab_sizes=vocab_sizes,
        cat_cols=DATA_CFG.categorical_features,
        num_numerical=len(DATA_CFG.numerical_features),
        text_embed_dim=DATA_CFG.text_embed_dim,
        image_feature_dim=DATA_CFG.image_feature_dim,
        tabular_hidden_dims=MODEL_CFG.tabular_hidden_dims,
        text_projection_dim=MODEL_CFG.text_projection_dim,
        image_projection_dim=MODEL_CFG.image_projection_dim,
        head_hidden_dims=MODEL_CFG.head_hidden_dims,
        dropout=MODEL_CFG.dropout,
        ghost_batch_size=MODEL_CFG.ghost_batch_size,
    )

    return model
