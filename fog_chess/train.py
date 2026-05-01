import copy
import os
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


from .chess import Minichess, FullChess, Move
from .networks import FogChessNet


def _make_game(board_size: int, max_moves: int, points_tiebreak: bool) -> Minichess:
    if board_size == 8:
        return FullChess(max_moves=max_moves, points_tiebreak=points_tiebreak)
    return Minichess(board_size=board_size, max_moves=max_moves, points_tiebreak=points_tiebreak)


def rollout_to_terminal_batch(
    games: List[Minichess],
    net: FogChessNet,
    for_player: int,
    max_depth: Optional[int] = None,
    copy_games: bool = True,
) -> List[float]:
    """Roll out multiple games in parallel, returning bootstrapped value estimates.

    At each visited state the value network provides an estimate from for_player's
    perspective. Terminal states use the actual game outcome. Returns the per-game
    average across all collected estimates.
    """
    if copy_games:
        games = [g.copy() for g in games]
    n = len(games)
    active = list(range(n))
    depths = [0] * n
    value_accumulators: List[List[float]] = [[] for _ in range(n)]

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
                need_move.append((idx, g, fog_log, moves, action_indices, current))

        for idx in finished:
            active.remove(idx)
            g = games[idx]
            if g.game_over:
                if g.winner == for_player:
                    outcome = 1.0
                elif g.winner == 3 - for_player:
                    outcome = -1.0
                else:
                    outcome = 0.0
                value_accumulators[idx].append(outcome)
            # depth-limited games rely solely on value estimates already collected

        if not need_move:
            break

        with torch.no_grad():
            outputs = net.forward_with_reconstruction_batch(
                [(fl, ai, current) for _, _, fl, _, ai, current in need_move]
            )

        for (idx, g, _, moves, _, current_player), out in zip(need_move, outputs):
            v = 2.0 * torch.sigmoid(out.value).item() - 1.0
            if current_player != for_player:
                v = -v
            value_accumulators[idx].append(v)

            action_idx = int(torch.multinomial(out.action_log_probs.exp(), 1).item())
            g.make_move(moves[action_idx])
            depths[idx] += 1

    return [sum(est) / len(est) if est else 0.0 for est in value_accumulators]


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

    with torch.no_grad():
        det_logs = net.reconstruct_masks_batch(fog_log, action_indices, k_samples, player)
        outputs = net.forward_batch([(det, action_indices, player) for det in det_logs])
    sampled = [
        int(torch.multinomial(out.action_log_probs.exp(), 1).item()) for out in outputs
    ]

    game_copies = [game.copy() for _ in range(k_samples)]
    for gc, action_idx in zip(game_copies, sampled):
        gc.make_move(moves[action_idx])

    outcomes = rollout_to_terminal_batch(game_copies, net, player, max_depth, copy_games=False)

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
    bool,  # whether MC search was run (policy loss only when True)
]


def pretrain_self_play_game(
    board_size: int = 5,
    max_moves: int = 100,
    points_tiebreak: bool = False,
) -> Tuple[List[Step], Minichess]:
    """Play one game with uniform random moves; only value and mask heads are trained."""
    game = _make_game(board_size, max_moves, points_tiebreak)
    trajectory: List[Step] = []
    bs = game.board_size

    while not game.game_over:
        player = game.current_player
        fog_log = game.get_fog_log(player)
        moves, action_indices = game.get_legal_moves_and_indices(player)
        if not moves:
            break

        true_mask_pieces: Dict[int, int] = {
            sq: game.board[sq // bs, sq % bs].value for _, sq, p in fog_log if p == 0
        }

        action_idx = random.randrange(len(moves))
        improved_probs = torch.ones(len(action_indices)) / len(action_indices)

        trajectory.append(
            (fog_log, action_indices, improved_probs, player, true_mask_pieces, False)
        )
        game.make_move(moves[action_idx])

    return trajectory, game


def self_play_game(
    rollout_net: FogChessNet,
    k_samples: int,
    mc_prob: float,
    mc_rollout_depth: Optional[int] = None,
    board_size: int = 5,
    max_moves: int = 100,
    points_tiebreak: bool = False,
) -> Tuple[List[Step], Minichess]:
    """Play one game with the EMA target network for both action selection and rollouts."""
    game = _make_game(board_size, max_moves, points_tiebreak)
    trajectory: List[Step] = []
    bs = game.board_size

    while not game.game_over:
        player = game.current_player
        fog_log = game.get_fog_log(player)
        moves, action_indices = game.get_legal_moves_and_indices(player)
        if not moves:
            break

        true_mask_pieces: Dict[int, int] = {
            sq: game.board[sq // bs, sq % bs].value for _, sq, p in fog_log if p == 0
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
                out = rollout_net.forward_with_reconstruction(fog_log, action_indices, player)
                action_idx = int(
                    torch.multinomial(out.action_log_probs.exp(), 1).item()
                )
            improved_probs = out.action_log_probs.exp()

        trajectory.append(
            (fog_log, action_indices, improved_probs, player, true_mask_pieces, use_mc)
        )

        game.make_move(moves[action_idx])

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
    device = next(net.parameters()).device

    for (
        fog_log,
        action_indices,
        improved_probs,
        player,
        true_mask_pieces,
        has_mc,
    ) in trajectory:
        out = net.forward(fog_log, action_indices, player)

        z = torch.tensor(player_outcome[player], device=device)
        loss_value = F.mse_loss(torch.sigmoid(out.value), (z + 1.0) / 2.0)
        value_losses.append(loss_value)

        if has_mc:
            loss_policy = -(improved_probs.detach().to(device) * out.action_log_probs).mean()
            policy_losses.append(loss_policy)

        mask_positions = [i + 1 for i, (_, _, p) in enumerate(fog_log) if p == 0]
        if mask_positions:
            mask_logits = net.mask_predictor(out.hidden[mask_positions])
            true_pieces = torch.tensor(
                [true_mask_pieces[sq] for _, sq, p in fog_log if p == 0],
                dtype=torch.long,
                device=device,
            )
            mask_losses.append(F.cross_entropy(mask_logits, true_pieces))

    mean_policy = (
        torch.stack(policy_losses).mean() if policy_losses else torch.tensor(0.0, device=device)
    )
    mean_value = torch.stack(value_losses).mean()
    mean_mask = torch.stack(mask_losses).mean() if mask_losses else torch.tensor(0.0, device=device)
    total = c_policy * mean_policy + c_value * mean_value + c_mask * mean_mask

    return total, mean_policy, mean_value, mean_mask


def save_checkpoint(
    path: str,
    net: FogChessNet,
    target_net: FogChessNet,
    optimizer,
    scheduler,
    phase: str,
    iteration: int,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "phase": phase,
            "iteration": iteration,
            "net": net.state_dict(),
            "target_net": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: str,
    net: FogChessNet,
    target_net: FogChessNet,
    optimizer,
    scheduler,
) -> Tuple[str, int]:
    ckpt = torch.load(path, weights_only=True)
    net.load_state_dict(ckpt["net"])
    target_net.load_state_dict(ckpt["target_net"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"[{_ts()}] Resumed from {path} (phase={ckpt['phase']}, iter={ckpt['iteration']})")
    return ckpt["phase"], ckpt["iteration"]


def update_ema(net: FogChessNet, target_net: FogChessNet, decay: float) -> None:
    with torch.no_grad():
        for p, p_t in zip(net.parameters(), target_net.parameters()):
            p_t.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def _update_step(
    net: FogChessNet,
    target_net: FogChessNet,
    optimizer,
    scheduler,
    replay_buffer: Deque,
    train_games_per_step: int,
    c_policy: float,
    c_value: float,
    c_mask: float,
    grad_clip: float,
    ema_decay: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Sample a batch from the replay buffer and perform one gradient update."""
    if len(replay_buffer) < train_games_per_step:
        return None
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
    return loss, loss_policy, loss_value, loss_mask


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


def evaluate_vs_greedy(
    net: FogChessNet,
    num_games: int,
    board_size: int = 5,
    max_moves: int = 100,
    points_tiebreak: bool = False,
) -> Dict[str, float]:
    """Play num_games against the greedy agent (alternating sides). Returns win/draw/loss rates."""
    wins = draws = losses = 0

    for game_idx in range(num_games):
        game = _make_game(board_size, max_moves, points_tiebreak)
        rl_player = 1 if game_idx % 2 == 0 else 2

        while not game.game_over:
            player = game.current_player
            if player == rl_player:
                fog_log = game.get_fog_log(player)
                moves, action_indices = game.get_legal_moves_and_indices(player)
                if not moves:
                    break
                with torch.no_grad():
                    out = net.forward_with_reconstruction(fog_log, action_indices, player)
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

    tcfg = cfg["train"]
    board_size: int = tcfg.get("board_size", 5)
    max_moves: int = tcfg.get("max_moves", 300 if board_size == 8 else 100)
    points_tiebreak: bool = tcfg.get("points_tiebreak", False)
    num_squares: int = board_size ** 2
    max_turns: int = max_moves + 1

    net = FogChessNet(**cfg["model"], num_squares=num_squares, max_turns=max_turns).to(device)
    target_net = copy.deepcopy(net)
    target_net.eval()

    optimizer = optim.Adam(net.parameters(), lr=tcfg["lr"])

    warmup_steps: int = tcfg.get("warmup_steps", 200)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
    )

    k_samples: int = tcfg["k_samples"]
    mc_prob: float = tcfg.get("mc_prob", 0.1)
    mc_rollout_depth: Optional[int] = tcfg.get("mc_rollout_depth", None)
    pretrain_iterations: int = tcfg.get("pretrain_iterations", 0)
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
    checkpoint_dir: str = tcfg.get("checkpoint_dir", "checkpoints/fog_chess")
    checkpoint_interval: int = tcfg.get("checkpoint_interval", 50)
    resume: Optional[str] = tcfg.get("resume", None)

    writer = SummaryWriter(log_dir=log_dir)
    replay_buffer: Deque[Tuple[List[Step], Minichess]] = deque(maxlen=buffer_size)

    resume_phase = "pretrain"
    resume_iter = -1
    if resume:
        resume_phase, resume_iter = load_checkpoint(
            resume, net, target_net, optimizer, scheduler
        )

    def _maybe_checkpoint(phase: str, iteration: int) -> None:
        if (iteration + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"{phase}_{iteration:06d}.pt")
            save_checkpoint(ckpt_path, net, target_net, optimizer, scheduler, phase, iteration)
            latest_path = os.path.join(checkpoint_dir, "latest.pt")
            save_checkpoint(latest_path, net, target_net, optimizer, scheduler, phase, iteration)
            print(f"[{_ts()}] Checkpoint saved: {ckpt_path}")

    # --- Pretraining phase ---
    pretrain_start = resume_iter + 1 if resume_phase == "pretrain" else pretrain_iterations
    for iteration in range(pretrain_start, pretrain_iterations):
        t0 = time.time()
        game_lengths = []
        for _ in range(games_per_update):
            trajectory, game = pretrain_self_play_game(board_size, max_moves, points_tiebreak)
            if trajectory:
                replay_buffer.append((trajectory, game))
                game_lengths.append(len(trajectory))

        print(
            f"[{_ts()}] pretrain {iteration}: self-play done in {time.time() - t0:.1f}s, buffer={len(replay_buffer)}",
            flush=True,
        )

        result = _update_step(
            net,
            target_net,
            optimizer,
            scheduler,
            replay_buffer,
            train_games_per_step,
            c_policy,
            c_value,
            c_mask,
            grad_clip,
            ema_decay,
        )
        if result is None:
            print(
                f"[{_ts()}] pretrain {iteration}: buffer too small ({len(replay_buffer)}<{train_games_per_step}), skipping update",
                flush=True,
            )
            continue

        loss, loss_policy, loss_value, loss_mask = result
        avg_game_len = sum(game_lengths) / len(game_lengths) if game_lengths else 0.0

        writer.add_scalar("pretrain/loss/total", loss.item(), iteration)
        writer.add_scalar("pretrain/loss/value", loss_value.item(), iteration)
        writer.add_scalar("pretrain/loss/mask", loss_mask.item(), iteration)
        writer.add_scalar("pretrain/game_length", avg_game_len, iteration)

        print(
            f"pretrain {iteration:5d} | loss {loss.item():8.4f} | "
            f"v {loss_value.item():.4f} | m {loss_mask.item():.4f} | "
            f"game_len {avg_game_len:.1f} | buffer {len(replay_buffer):4d}"
        )
        _maybe_checkpoint("pretrain", iteration)

    # --- Self-play training phase ---
    train_start = resume_iter + 1 if resume_phase == "train" else 0
    for iteration in range(train_start, num_iterations):
        t0 = time.time()
        game_lengths = []
        for _ in range(games_per_update):
            trajectory, game = self_play_game(
                target_net, k_samples, mc_prob, mc_rollout_depth,
                board_size, max_moves, points_tiebreak,
            )
            if trajectory:
                replay_buffer.append((trajectory, game))
                game_lengths.append(len(trajectory))

        print(
            f"[{_ts()}] iter {iteration}: self-play done in {time.time() - t0:.1f}s, buffer={len(replay_buffer)}",
            flush=True,
        )

        result = _update_step(
            net,
            target_net,
            optimizer,
            scheduler,
            replay_buffer,
            train_games_per_step,
            c_policy,
            c_value,
            c_mask,
            grad_clip,
            ema_decay,
        )
        if result is None:
            print(
                f"[{_ts()}] iter {iteration}: buffer too small ({len(replay_buffer)}<{train_games_per_step}), skipping update",
                flush=True,
            )
            continue

        loss, loss_policy, loss_value, loss_mask = result
        avg_game_len = sum(game_lengths) / len(game_lengths) if game_lengths else 0.0
        lr = scheduler.get_last_lr()[0]

        writer.add_scalar("loss/total", loss.item(), iteration)
        writer.add_scalar("loss/policy", loss_policy.item(), iteration)
        writer.add_scalar("loss/value", loss_value.item(), iteration)
        writer.add_scalar("loss/mask", loss_mask.item(), iteration)
        writer.add_scalar("train/lr", lr, iteration)
        writer.add_scalar("train/buffer_size", len(replay_buffer), iteration)
        writer.add_scalar("train/game_length", avg_game_len, iteration)

        print(
            f"iter {iteration:5d} | loss {loss.item():8.4f} | "
            f"p {loss_policy.item():.4f} | v {loss_value.item():.4f} | "
            f"m {loss_mask.item():.4f} | game_len {avg_game_len:.1f} | "
            f"buffer {len(replay_buffer):4d} | lr {lr:.2e}"
        )

        if (iteration + 1) % eval_interval == 0:
            net.eval()
            stats = evaluate_vs_greedy(net, eval_games, board_size, max_moves, points_tiebreak)
            net.train()

            writer.add_scalar("eval/win_rate_vs_greedy", stats["win_rate"], iteration)
            writer.add_scalar("eval/draw_rate_vs_greedy", stats["draw_rate"], iteration)
            writer.add_scalar("eval/loss_rate_vs_greedy", stats["loss_rate"], iteration)

            print(
                f"  [eval vs greedy] win={stats['win_rate']:.2%} "
                f"draw={stats['draw_rate']:.2%} "
                f"loss={stats['loss_rate']:.2%}"
            )
        _maybe_checkpoint("train", iteration)

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
