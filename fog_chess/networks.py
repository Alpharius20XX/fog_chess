"""PyTorch networks for fog-of-war mini chess."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple

# Vocabulary constants matching chess.Piece enum (0=UNKNOWN … 13=BLACK_KING)
NUM_PIECES = 14
NUM_SQUARES = 25  # 5×5 board
MAX_TURNS = 101  # turns 0–100

# Token-type IDs
_TYPE_CLS = 0
_TYPE_FOG_KNOWN = 1
_TYPE_FOG_MASK = 2
_TYPE_ACTION = 3


class StateEncoder(nn.Module):
    """Transformer over a fog-log + action token sequence.

    Sequence layout: [CLS] [fog_0] … [fog_N] [action_0] … [action_M]

    Each fog entry with piece_value == 0 (UNKNOWN) becomes a MASK token;
    all other fog entries embed their (turn, square, piece) triple.
    Action entries embed their (start_sq, end_sq) pair.
    All embeddings are summed with a token-type embedding.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_head: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.cls_token = nn.Parameter(torch.randn(d_model))
        self.mask_token = nn.Parameter(torch.randn(d_model))

        self.turn_emb = nn.Embedding(MAX_TURNS, d_model)
        self.square_emb = nn.Embedding(NUM_SQUARES, d_model)
        self.piece_emb = nn.Embedding(NUM_PIECES, d_model)
        self.start_sq_emb = nn.Embedding(NUM_SQUARES, d_model)
        self.end_sq_emb = nn.Embedding(NUM_SQUARES, d_model)
        self.token_type_emb = nn.Embedding(4, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )

    def embed(
        self,
        fog_log: List[Tuple[int, int, int]],
        action_indices: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, List[int]]:
        """Build the full token sequence with batched embedding lookups.

        Returns:
          tokens: [seq_len, d_model]
          mask_positions: sequence indices (1-based, after CLS) of MASK tokens
        """
        device = self.cls_token.device
        n_fog = len(fog_log)
        n_act = len(action_indices)
        seq_len = 1 + n_fog + n_act

        tokens = torch.empty(seq_len, self.d_model, device=device)

        # CLS token
        tokens[0] = self.cls_token + self.token_type_emb.weight[_TYPE_CLS]

        mask_positions: List[int] = []

        if n_fog > 0:
            turns = torch.tensor([t for t, _, _ in fog_log], device=device, dtype=torch.long)
            sqs = torch.tensor([s for _, s, _ in fog_log], device=device, dtype=torch.long)
            pieces = torch.tensor([p for _, _, p in fog_log], device=device, dtype=torch.long)

            is_mask = pieces == 0
            is_known = ~is_mask

            base = self.turn_emb(turns) + self.square_emb(sqs)  # [n_fog, d_model]

            fog_tokens = torch.empty(n_fog, self.d_model, device=device)
            if is_known.any():
                fog_tokens[is_known] = (
                    self.piece_emb(pieces[is_known])
                    + base[is_known]
                    + self.token_type_emb.weight[_TYPE_FOG_KNOWN]
                )
            if is_mask.any():
                fog_tokens[is_mask] = (
                    self.mask_token
                    + base[is_mask]
                    + self.token_type_emb.weight[_TYPE_FOG_MASK]
                )

            tokens[1 : 1 + n_fog] = fog_tokens
            mask_positions = (is_mask.nonzero(as_tuple=True)[0] + 1).tolist()

        if n_act > 0:
            starts = torch.tensor([s for s, _ in action_indices], device=device, dtype=torch.long)
            ends = torch.tensor([e for _, e in action_indices], device=device, dtype=torch.long)
            tokens[1 + n_fog :] = (
                self.start_sq_emb(starts)
                + self.end_sq_emb(ends)
                + self.token_type_emb.weight[_TYPE_ACTION]
            )

        return tokens, mask_positions

    def forward(self, token_seq: torch.Tensor) -> torch.Tensor:
        """[batch, seq_len, d_model] → [batch, seq_len, d_model]"""
        return self.transformer(token_seq)


class MaskPredictor(nn.Module):
    """Predicts piece type at a MASK position from the encoder hidden state."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.linear = nn.Linear(d_model, NUM_PIECES)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [..., d_model] → [..., NUM_PIECES] logits"""
        return self.linear(hidden)


class ValueHead(nn.Module):
    """Predicts game outcome from the CLS hidden state."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        """cls_hidden: [batch, d_model] → [batch] logits (sigmoid gives win probability)"""
        return self.mlp(cls_hidden).squeeze(-1)


class PolicyHead(nn.Module):
    """Scores action tokens and returns a log-probability distribution."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, action_hiddens: torch.Tensor) -> torch.Tensor:
        """action_hiddens: [batch, num_actions, d_model] → [batch, num_actions] log-probs"""
        logits = self.scorer(action_hiddens).squeeze(-1)
        return F.log_softmax(logits, dim=-1)


@dataclass
class ForwardOutput:
    value: torch.Tensor  # scalar — logit; sigmoid gives win probability
    action_log_probs: torch.Tensor  # [num_actions] log-probabilities over legal moves
    hidden: torch.Tensor  # [seq_len, d_model] full encoder output


class FogChessNet(nn.Module):
    """Complete fog-of-war chess network.

    Typical inference flow:
      determinized = net.reconstruct_masks(fog_log, actions)
      out = net(determinized, actions)

    Typical training:
      - Mask-reconstruction loss: cross-entropy at MASK positions against true pieces.
        out = net(fog_log, actions)       # fog_log may contain UNKNOWNs
        _, mask_positions = net.encoder.embed(fog_log, actions)
        mask_logits = net.mask_predictor(out.hidden[mask_positions])
        loss_mask = F.cross_entropy(mask_logits, true_piece_ids)
      - Value loss: MSE or BCE between out.value and game outcome.
      - Policy loss: REINFORCE or PPO using out.action_log_probs.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_head: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = StateEncoder(d_model, n_head, n_layers, dim_feedforward, dropout)
        self.mask_predictor = MaskPredictor(d_model)
        self.value_head = ValueHead(d_model)
        self.policy_head = PolicyHead(d_model)

    def forward(
        self,
        fog_log: List[Tuple[int, int, int]],
        action_indices: List[Tuple[int, int]],
    ) -> ForwardOutput:
        """Forward pass over the fog log (UNKNOWN entries become MASK tokens).

        Args:
          fog_log: [(turn, square, piece_value), ...] from Minichess.get_fog_log()
          action_indices: [(start_sq, end_sq), ...] from Minichess.get_legal_move_indices()
        """
        n_fog = len(fog_log)
        tokens, _ = self.encoder.embed(fog_log, action_indices)
        hidden = self.encoder(tokens.unsqueeze(0))  # [1, seq_len, d_model]

        cls_h = hidden[:, 0, :]  # [1, d_model]
        action_h = hidden[:, 1 + n_fog :, :]  # [1, n_actions, d_model]

        value = self.value_head(cls_h).squeeze(0)
        action_log_probs = self.policy_head(action_h).squeeze(0)

        return ForwardOutput(
            value=value,
            action_log_probs=action_log_probs,
            hidden=hidden.squeeze(0),
        )

    @torch.no_grad()
    def reconstruct_masks(
        self,
        fog_log: List[Tuple[int, int, int]],
        action_indices: List[Tuple[int, int]],
        temperature: float = 1.0,
    ) -> List[Tuple[int, int, int]]:
        """Single-pass mask reconstruction: one forward pass samples all unknowns at once.

        Each MASK token's predicted distribution is sampled independently from a
        single encoder pass, replacing the O(N²) autoregressive approach with O(N).
        """
        fog_log = list(fog_log)
        pending = [i for i, (_, _, p) in enumerate(fog_log) if p == 0]
        if not pending:
            return fog_log

        tokens, _ = self.encoder.embed(fog_log, action_indices)
        hidden = self.encoder(tokens.unsqueeze(0))[0]  # [seq_len, d_model]

        for fog_idx in pending:
            logits = self.mask_predictor(hidden[fog_idx + 1]) / temperature
            probs = F.softmax(logits, dim=-1)
            sampled = int(torch.multinomial(probs, 1).item())
            turn, sq, _ = fog_log[fog_idx]
            fog_log[fog_idx] = (turn, sq, sampled)

        return fog_log

    def forward_with_reconstruction(
        self,
        fog_log: List[Tuple[int, int, int]],
        action_indices: List[Tuple[int, int]],
        temperature: float = 1.0,
    ) -> ForwardOutput:
        """Reconstruct all MASK tokens in one pass, then run the full forward pass."""
        determinized = self.reconstruct_masks(fog_log, action_indices, temperature)
        return self.forward(determinized, action_indices)
