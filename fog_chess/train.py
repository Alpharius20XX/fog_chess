"""Training loop for fog-of-war chess.

At each step, run k_samples rollouts to estimate per-action returns.
Each rollout:
1. Sample an action from the policy via forward_with_reconstruction (internal
   determinization + forward) on the current player's fog log.
2. Apply that action to a copy of the true game state.
3. Roll out to terminal; both players use forward_with_reconstruction each turn.
Training targets:
- Policy: softmax of per-action average MC returns.
- Value: game outcome (±1 or 0).
- Mask: true piece type at UNKNOWN positions.
"""

import copy
import random
import time
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from typing import Deque, Dict, List, Optional, Tuple


def _ts() -> str:
    return time.strftime("%H:%M:%S")


from .chess import Minichess, Move
from .networks import FogChessNet


def rollout_to_terminal_batch(
    games: List[Minichess],
    net: FogChessNet,
    for_player: int,
    max_depth: Optional[int] = None,
) -> List[float]:
    """Roll out multiple games in parallel using batched inference.

    Stops at terminal or after max_depth turns (per game). When capped by
    depth, the outcome is 0 (draw) for unfinished games.
    """
    games = [g.copy() for g in games]
    n = len(games)
    outcomes = [0.0] * n
    active = list(range(n))
    depths = [0] * n

    while active:
        need_move = []
        finished = []

        for idx in active:
            g = games[idx]
            if g.game_over:
                finished.append(idx)
                continue
            if max_depth is not None and depths[idx] >= max_depth:
                finished.append(idx)
                continue
            current = g.current_player
            fog_log = g.get_fog_log(current)
            moves, action_indices = g.get_legal_moves_and_indices(current)
            if not moves:
                finished.append(idx)
            else:
                need_move.append((idx, g, fog_log, moves, action_indices))

        for idx in finished:
            active.remove(idx)
            g = games[idx]
            if g.winner == for_player:
                outcomes[idx] = 1.0
            elif g.winner == 3 - for_player:
                outcomes[idx] = -1.0

        if not need_move:
            break

        with torch.no_grad():
            outputs = net.forward_with_reconstruction_batch(
                [(fl, ai) for _, _, fl, _, ai in need_move]
            )

        for (idx, g, _, moves, _), out in zip(need_move, outputs):
            action_idx = int(torch.multinomial(out.action_log_probs.exp(), 1).item())
            g.make_move(moves[action_idx])
            depths[idx] += 1

    return outcomes


def mc_action_search(
    game: Minichess,
    net: FogChessNet,
    fog_log: List[Tuple[int, int, int]],
    moves: List[Move],
    action_indices: List[Tuple[int, int]],
    k_samples: int,
    player: int,
    max_depth: Optional[int] = None,
) -> torch.Tensor:
    """Run k_samples MC rollouts with batched inference."""
    n_actions = len(action_indices)

    # One encoder pass samples k_samples determinizations; one batched forward scores them.
    with torch.no_grad():
        det_logs = net.reconstruct_masks_batch(fog_log, action_indices, k_samples)
        outputs = net.forward_batch([(det, action_indices) for det in det_logs])
    sampled = [
        int(torch.multinomial(out.action_log_probs.exp(), 1).item()) for out in outputs
    ]

    game_copies = [game.copy() for _ in range(k_samples)]
    for gc, action_idx in zip(game_copies, sampled):
        gc.make_move(moves[action_idx])

    outcomes = rollout_to_terminal_batch(game_copies, net, player, max_depth)

    action_returns = torch.zeros(n_actions)
    action_counts = torch.zeros(n_actions)
    for action_idx, outcome in zip(sampled, outcomes):
        action_returns[action_idx] += outcome
        action_counts[action_idx] += 1

    avg_returns = torch.where(
        action_counts > 0,
        action_returns / action_counts.clamp(min=1),
        torch.full((n_actions,), -1.0),
    )
    return F.softmax(avg_returns, dim=0)


Step = Tuple[
    List[Tuple[int, int, int]],  # fog_log
    List[Tuple[int, int]],  # action_indices
    torch.Tensor,  # improved_probs [n_actions]
    int,  # player (1 or 2)
    Dict[int, int],  # sq -> true piece.value for each UNKNOWN square
    bool,  # whether MC search was run (policy loss only applied when True)
]


def self_play_game(
    rollout_net: FogChessNet,
    k_samples: int,
    mc_prob: float,
    mc_rollout_depth: Optional[int] = None,
) -> Tuple[List[Step], Minichess]:
    """Play one game with the EMA target network for both action selection and rollouts."""
    game = Minichess()
    trajectory: List[Step] = []

    move_num = 0
    while not game.game_over:
        player = game.current_player
        fog_log = game.get_fog_log(player)
        moves, action_indices = game.get_legal_moves_and_indices(player)
        if not moves:
            break

        true_mask_pieces: Dict[int, int] = {
            sq: game.board[sq // 5, sq % 5].value for _, sq, p in fog_log if p == 0
        }

        use_mc = random.random() < mc_prob
        if use_mc:
            improved_probs = mc_action_search(
                game,
                rollout_net,
                fog_log,
                moves,
                action_indices,
                k_samples,
                player,
                mc_rollout_depth,
            )
            action_idx = int(torch.multinomial(improved_probs, 1).item())
        else:
            with torch.no_grad():
                out = rollout_net.forward_with_reconstruction(fog_log, action_indices)
                action_idx = int(
                    torch.multinomial(out.action_log_probs.exp(), 1).item()
                )
            improved_probs = out.action_log_probs.exp()

        trajectory.append(
            (fog_log, action_indices, improved_probs, player, true_mask_pieces, use_mc)
        )

        game.make_move(moves[action_idx])
        move_num += 1

    return trajectory, game


def compute_losses(
    net: FogChessNet,
    trajectory: List[Step],
    game: Minichess,
    c_policy: float = 1.0,
    c_value: float = 1.0,
    c_mask: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total, policy, value, mask) losses averaged over the trajectory."""
    if game.winner == 1:
        player_outcome = {1: 1.0, 2: -1.0}
    elif game.winner == 2:
        player_outcome = {1: -1.0, 2: 1.0}
    else:
        player_outcome = {1: 0.0, 2: 0.0}

    policy_losses: List[torch.Tensor] = []
    value_losses: List[torch.Tensor] = []
    mask_losses: List[torch.Tensor] = []

    for (
        fog_log,
        action_indices,
        improved_probs,
        player,
        true_mask_pieces,
        has_mc,
    ) in trajectory:
        out = net.forward(fog_log, action_indices)

        z = torch.tensor(player_outcome[player])
        loss_value = F.mse_loss(torch.sigmoid(out.value), (z + 1.0) / 2.0)
        value_losses.append(loss_value)

        if has_mc:
            loss_policy = -(improved_probs.detach() * out.action_log_probs).mean()
            policy_losses.append(loss_policy)

        mask_positions = [i + 1 for i, (_, _, p) in enumerate(fog_log) if p == 0]
        if mask_positions:
            mask_logits = net.mask_predictor(out.hidden[mask_positions])
            true_pieces = torch.tensor(
                [true_mask_pieces[sq] for _, sq, p in fog_log if p == 0],
                dtype=torch.long,
            )
            mask_losses.append(F.cross_entropy(mask_logits, true_pieces))

    mean_policy = (
        torch.stack(policy_losses).mean() if policy_losses else torch.tensor(0.0)
    )
    mean_value = torch.stack(value_losses).mean()
    mean_mask = torch.stack(mask_losses).mean() if mask_losses else torch.tensor(0.0)
    total = c_policy * mean_policy + c_value * mean_value + c_mask * mean_mask

    return total, mean_policy, mean_value, mean_mask


def update_ema(net: FogChessNet, target_net: FogChessNet, decay: float) -> None:
    with torch.no_grad():
        for p, p_t in zip(net.parameters(), target_net.parameters()):
            p_t.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def greedy_agent_move(game: Minichess) -> Optional[Move]:
    """Captures the highest-value enemy piece visible; otherwise plays randomly."""
    moves = game.get_all_valid_moves()
    if not moves:
        return None
    captures = [m for m in moves if m.captured is not None]
    if captures:
        return max(
            captures, key=lambda m: abs(m.captured.get_value(True)) if m.captured else 0
        )
    return random.choice(moves)


def evaluate_vs_greedy(net: FogChessNet, num_games: int) -> Dict[str, float]:
    """Play num_games against the greedy agent (alternating sides). Returns win/draw/loss rates."""
    wins = draws = losses = 0

    for game_idx in range(num_games):
        game = Minichess()
        rl_player = 1 if game_idx % 2 == 0 else 2

        while not game.game_over:
            player = game.current_player
            if player == rl_player:
                fog_log = game.get_fog_log(player)
                moves, action_indices = game.get_legal_moves_and_indices(player)
                if not moves:
                    break
                with torch.no_grad():
                    out = net.forward_with_reconstruction(fog_log, action_indices)
                    action_idx = int(
                        torch.multinomial(out.action_log_probs.exp(), 1).item()
                    )
                game.make_move(moves[action_idx])
            else:
                move = greedy_agent_move(game)
                if move is None:
                    break
                game.make_move(move)

        if game.winner == rl_player:
            wins += 1
        elif game.winner == 0:
            draws += 1
        else:
            losses += 1

    return {
        "win_rate": wins / num_games,
        "draw_rate": draws / num_games,
        "loss_rate": losses / num_games,
    }


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FogChessNet(**cfg["model"]).to(device)
    target_net = copy.deepcopy(net)
    target_net.eval()

    tcfg = cfg["train"]
    optimizer = optim.Adam(net.parameters(), lr=tcfg["lr"])

    warmup_steps: int = tcfg.get("warmup_steps", 200)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
    )

    k_samples: int = tcfg["k_samples"]
    mc_prob: float = tcfg.get("mc_prob", 0.1)
    mc_rollout_depth: Optional[int] = tcfg.get("mc_rollout_depth", None)
    num_iterations: int = tcfg.get("num_iterations", 1000)
    ema_decay: float = tcfg.get("target_ema", 0.995)
    grad_clip: float = tcfg.get("grad_clip", 1.0)
    games_per_update: int = tcfg.get("games_per_update", 4)
    buffer_size: int = tcfg.get("buffer_size", 128)
    train_games_per_step: int = tcfg.get("train_games_per_step", 8)
    c_policy: float = tcfg.get("c_policy", 1.0)
    c_value: float = tcfg.get("c_value", 1.0)
    c_mask: float = tcfg.get("c_mask", 1.0)
    eval_interval: int = tcfg.get("eval_interval", 50)
    eval_games: int = tcfg.get("eval_games", 20)
    log_dir: str = tcfg.get("log_dir", "runs/fog_chess")

    writer = SummaryWriter(log_dir=log_dir)
    replay_buffer: Deque[Tuple[List[Step], Minichess]] = deque(maxlen=buffer_size)

    for iteration in range(num_iterations):
        t0 = time.time()
        for g_idx in range(games_per_update):
            trajectory, game = self_play_game(
                target_net, k_samples, mc_prob, mc_rollout_depth
            )
            if trajectory:
                replay_buffer.append((trajectory, game))
        print(
            f"[{_ts()}] iter {iteration}: self-play done in {time.time() - t0:.1f}s, buffer={len(replay_buffer)}",
            flush=True,
        )

        if len(replay_buffer) < train_games_per_step:
            print(
                f"[{_ts()}] iter {iteration}: buffer too small ({len(replay_buffer)}<{train_games_per_step}), skipping update",
                flush=True,
            )
            continue

        print(
            f"[{_ts()}] iter {iteration}: training on batch of {train_games_per_step}",
            flush=True,
        )
        batch = random.sample(list(replay_buffer), train_games_per_step)

        optimizer.zero_grad()
        results = [
            compute_losses(net, traj, g, c_policy, c_value, c_mask) for traj, g in batch
        ]
        totals, policies, values, masks = zip(*results)

        loss = torch.stack(totals).mean()
        loss_policy = torch.stack(policies).mean()
        loss_value = torch.stack(values).mean()
        loss_mask = torch.stack(masks).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        update_ema(net, target_net, ema_decay)

        lr = scheduler.get_last_lr()[0]
        writer.add_scalar("loss/total", loss.item(), iteration)
        writer.add_scalar("loss/policy", loss_policy.item(), iteration)
        writer.add_scalar("loss/value", loss_value.item(), iteration)
        writer.add_scalar("loss/mask", loss_mask.item(), iteration)
        writer.add_scalar("train/lr", lr, iteration)
        writer.add_scalar("train/buffer_size", len(replay_buffer), iteration)

        print(
            f"iter {iteration:5d} | loss {loss.item():8.4f} | "
            f"p {loss_policy.item():.4f} | v {loss_value.item():.4f} | "
            f"m {loss_mask.item():.4f} | buffer {len(replay_buffer):4d} | "
            f"lr {lr:.2e}"
        )

        if (iteration + 1) % eval_interval == 0:
            net.eval()
            stats = evaluate_vs_greedy(net, eval_games)
            net.train()

            writer.add_scalar("eval/win_rate_vs_greedy", stats["win_rate"], iteration)
            writer.add_scalar("eval/draw_rate_vs_greedy", stats["draw_rate"], iteration)
            writer.add_scalar("eval/loss_rate_vs_greedy", stats["loss_rate"], iteration)

            print(
                f"  [eval vs greedy] win={stats['win_rate']:.2%} "
                f"draw={stats['draw_rate']:.2%} "
                f"loss={stats['loss_rate']:.2%}"
            )

    writer.close()


def main(cfg: dict):
    train(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
