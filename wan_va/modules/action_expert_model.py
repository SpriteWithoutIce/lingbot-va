import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionExpertBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.self_ln = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_ln_1 = nn.LayerNorm(hidden_dim)
        self.cross_attn_1 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_ln_2 = nn.LayerNorm(hidden_dim)
        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_ln_3 = nn.LayerNorm(hidden_dim)
        self.cross_attn_3 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_ln_state = nn.LayerNorm(hidden_dim)
        self.cross_attn_state = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn_ln = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim),
        )

        # Gate(state) = sigmoid((threshold - |delta_v|) * temperature + bias)
        self.gate_threshold = nn.Parameter(torch.tensor(0.10))
        self.gate_temperature_raw = nn.Parameter(torch.tensor(1.0))
        self.gate_bias = nn.Parameter(torch.tensor(0.0))

    def _cross(self, x, cond, ln, attn, key_padding_mask=None, attn_mask=None):
        q = ln(x)
        out, _ = attn(
            q,
            cond,
            cond,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return x + out

    def forward(
        self,
        x,
        vel_cond,
        state_cond,
        vel_magnitude,
        query_padding_mask=None,
        condition_padding_mask=None,
        causal_attn_mask=None,
    ):
        q = self.self_ln(x)
        self_out, _ = self.self_attn(
            q,
            q,
            q,
            key_padding_mask=query_padding_mask,
            attn_mask=causal_attn_mask,
            need_weights=False,
        )
        x = x + self_out

        x = self._cross(
            x,
            vel_cond,
            self.cross_ln_1,
            self.cross_attn_1,
            key_padding_mask=condition_padding_mask,
            attn_mask=causal_attn_mask,
        )
        x = self._cross(
            x,
            vel_cond,
            self.cross_ln_2,
            self.cross_attn_2,
            key_padding_mask=condition_padding_mask,
            attn_mask=causal_attn_mask,
        )
        x = self._cross(
            x,
            vel_cond,
            self.cross_ln_3,
            self.cross_attn_3,
            key_padding_mask=condition_padding_mask,
            attn_mask=causal_attn_mask,
        )

        q_state = self.cross_ln_state(x)
        state_out, _ = self.cross_attn_state(
            q_state,
            state_cond,
            state_cond,
            key_padding_mask=condition_padding_mask,
            attn_mask=causal_attn_mask,
            need_weights=False,
        )
        gate_temperature = F.softplus(self.gate_temperature_raw) + 1e-4
        gate = torch.sigmoid(
            (self.gate_threshold - vel_magnitude) * gate_temperature + self.gate_bias
        )
        x = x + gate * state_out

        x = x + self.ffn(self.ffn_ln(x))
        return x, gate


class FlowMatchingActionExpert(nn.Module):
    def __init__(
        self,
        action_dim,
        state_dim,
        velocity_dim=None,
        hidden_dim=768,
        num_heads=12,
        num_layers=8,
        dropout=0.1,
        ffn_mult=4,
        timestep_dim=256,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.velocity_dim = action_dim if velocity_dim is None else velocity_dim
        self.hidden_dim = hidden_dim

        self.noisy_action_proj = nn.Linear(action_dim, hidden_dim)
        self.vel_cond_proj = nn.Linear(self.velocity_dim, hidden_dim)
        self.state_cond_proj = nn.Linear(state_dim, hidden_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(1, timestep_dim),
            nn.SiLU(),
            nn.Linear(timestep_dim, hidden_dim),
        )
        self.action_pos_embed = nn.Embedding(2048, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                ActionExpertBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.to_action = nn.Linear(hidden_dim, action_dim)

    def _build_block_causal_mask(self, seq_len, block_size, device):
        token_ids = torch.arange(seq_len, device=device)
        q_block = token_ids[:, None] // block_size
        kv_block = token_ids[None, :] // block_size
        allow = kv_block <= q_block
        # For nn.MultiheadAttention bool mask: True means masked(disallowed).
        return ~allow

    def _build_dual_stream_block_mask(self, seq_len, block_size, device):
        # Token order: [noisy(L), clean(L)].
        token_ids = torch.arange(seq_len, device=device)
        stream_id = token_ids // (seq_len // 2)  # 0: noisy, 1: clean
        local_id = token_ids % (seq_len // 2)
        q_block = (local_id[:, None] // block_size)
        kv_block = (local_id[None, :] // block_size)
        q_stream = stream_id[:, None]
        kv_stream = stream_id[None, :]

        clean_to_clean = (q_stream == 1) & (kv_stream == 1) & (kv_block <= q_block)
        noisy_to_clean = (q_stream == 0) & (kv_stream == 1) & (kv_block < q_block)
        noisy_to_noisy = (q_stream == 0) & (kv_stream == 0) & (kv_block == q_block)
        allow = clean_to_clean | noisy_to_clean | noisy_to_noisy
        return ~allow

    def _flatten_actions(self, actions):
        b, c, f, n, _ = actions.shape
        seq = actions.permute(0, 2, 3, 1, 4).reshape(b, f * n, c)
        return seq, (f, n)

    def _flatten_states(self, states, target_f, target_n):
        # states: [B, F, N, C] -> [B, F*N, C]
        if states.dim() != 4:
            raise ValueError(f"Expected states shape [B, F, N, C], got {tuple(states.shape)}")
        b, f, n, c = states.shape
        if f != target_f or n != target_n:
            raise ValueError(
                f"State/action temporal mismatch: states {(f, n)} vs actions {(target_f, target_n)}"
            )
        return states.reshape(b, f * n, c)

    def _build_velocity_condition(self, latent_velocity, target_f, target_n):
        # latent_velocity: [B, C, F] pooled from WAN latent velocity field.
        vel_diff = latent_velocity[:, :, 1:] - latent_velocity[:, :, :-1]  # [B, C, F-1]
        b, c, f_minus_1 = vel_diff.shape
        aligned = torch.zeros(b, c, target_f, device=vel_diff.device, dtype=vel_diff.dtype)
        copy_f = min(target_f, f_minus_1)
        if copy_f > 0:
            aligned[:, :, :copy_f] = vel_diff[:, :, :copy_f]
        cond = aligned.permute(0, 2, 1).unsqueeze(2).repeat(1, 1, target_n, 1)
        cond = cond.reshape(b, target_f * target_n, c)
        vel_mag = torch.norm(cond, dim=-1, keepdim=True)  # [B, F*N, 1]
        return cond, vel_mag

    def forward(
        self,
        noisy_actions,
        timesteps,
        latent_velocity,
        robot_states,
        action_padding_mask=None,
        clean_actions=None,
    ):
        action_seq, (f, n) = self._flatten_actions(noisy_actions)
        state_seq = self._flatten_states(robot_states, target_f=f, target_n=n)
        vel_cond, vel_mag = self._build_velocity_condition(latent_velocity, target_f=f, target_n=n)
        x_noisy = self.noisy_action_proj(action_seq)
        x_noisy = x_noisy + self.time_embed(timesteps.unsqueeze(-1).to(x_noisy.dtype))
        pos_ids = torch.arange(x_noisy.shape[1], device=x_noisy.device)
        x_noisy = x_noisy + self.action_pos_embed(pos_ids)[None]

        vel_cond_noisy = self.vel_cond_proj(vel_cond)
        state_cond_noisy = self.state_cond_proj(state_seq)

        if clean_actions is not None:
            clean_seq, _ = self._flatten_actions(clean_actions)
            x_clean = self.noisy_action_proj(clean_seq)
            t_clean = torch.zeros_like(timesteps)
            x_clean = x_clean + self.time_embed(t_clean.unsqueeze(-1).to(x_clean.dtype))
            x_clean = x_clean + self.action_pos_embed(pos_ids)[None]

            vel_cond_clean = self.vel_cond_proj(vel_cond)
            state_cond_clean = self.state_cond_proj(state_seq)

            x = torch.cat([x_noisy, x_clean], dim=1)
            vel_cond_full = torch.cat([vel_cond_noisy, vel_cond_clean], dim=1)
            state_cond_full = torch.cat([state_cond_noisy, state_cond_clean], dim=1)
            vel_mag_full = torch.cat([vel_mag, vel_mag], dim=1)
            if action_padding_mask is not None:
                query_padding_mask = torch.cat(
                    [action_padding_mask, action_padding_mask], dim=1
                )
                condition_padding_mask = query_padding_mask
            else:
                query_padding_mask = None
                condition_padding_mask = None
            causal_attn_mask = self._build_dual_stream_block_mask(
                seq_len=x.shape[1], block_size=n, device=x.device
            )
            noisy_len = x_noisy.shape[1]
        else:
            x = x_noisy
            vel_cond_full = vel_cond_noisy
            state_cond_full = state_cond_noisy
            vel_mag_full = vel_mag
            query_padding_mask = action_padding_mask
            condition_padding_mask = action_padding_mask
            causal_attn_mask = self._build_block_causal_mask(
                seq_len=x.shape[1], block_size=n, device=x.device
            )
            noisy_len = x.shape[1]

        gate_values = []
        for block in self.blocks:
            x, gate = block(
                x,
                vel_cond=vel_cond_full,
                state_cond=state_cond_full,
                vel_magnitude=vel_mag_full,
                query_padding_mask=query_padding_mask,
                condition_padding_mask=condition_padding_mask,
                causal_attn_mask=causal_attn_mask,
            )
            gate_values.append(gate)

        x_noisy_out = x[:, :noisy_len]
        pred_action = self.to_action(self.norm_out(x_noisy_out))
        pred_action = pred_action.view(pred_action.shape[0], f, n, self.action_dim)
        mean_gate = torch.stack(gate_values, dim=0).mean()
        return pred_action, mean_gate
