from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Tuple

from .chess import Minichess, Move, Piece

ObservationKey = Tuple[int, ...]
ActionKey = str
QTable = Dict[ObservationKey, Dict[ActionKey, float]]
MemoryState = Tuple[Tuple[Piece, ...], ...]


def observation_key(observation: Tuple[Tuple[Piece, ...], ...]) -> ObservationKey:
    return tuple(piece.value for row in observation for piece in row)


def update_memory(
    memory: MemoryState | None, observation: Tuple[Tuple[Piece, ...], ...]
) -> MemoryState:
    if memory is None:
        memory = tuple(tuple(Piece.UNKNOWN for _ in range(5)) for _ in range(5))

    rows = []
    for row_idx, row in enumerate(observation):
        memory_row = memory[row_idx]
        rows.append(
            tuple(
                piece if piece != Piece.UNKNOWN else memory_row[col_idx]
                for col_idx, piece in enumerate(row)
            )
        )
    return tuple(rows)


def action_key(move: Move) -> ActionKey:
    return move.to_notation()


def _max_q(q_table: QTable, state: ObservationKey, legal_moves: Iterable[Move]) -> float:
    values = q_table.get(state, {})
    legal_keys = [action_key(move) for move in legal_moves]
    if not legal_keys:
        return 0.0
    return max(values.get(key, 0.0) for key in legal_keys)


def _choose_action(
    q_table: QTable,
    state: ObservationKey,
    legal_moves: List[Move],
    epsilon: float,
    rng: random.Random,
) -> Move:
    if rng.random() < epsilon:
        return rng.choice(legal_moves)

    values = q_table.get(state, {})
    best_value = max(values.get(action_key(move), 0.0) for move in legal_moves)
    best_moves = [
        move for move in legal_moves if values.get(action_key(move), 0.0) == best_value
    ]
    return rng.choice(best_moves)


def choose_action_for_game(
    q_tables: MutableMapping[int, QTable],
    game: Minichess,
    rng: random.Random,
    epsilon: float = 0.0,
    memory: MemoryState | None = None,
) -> Move:
    player = game.current_player
    legal_moves = game.get_all_valid_moves()
    if not legal_moves:
        raise ValueError("No legal moves available.")
    observation = game.get_observation(player)
    state = observation_key(update_memory(memory, observation))
    return _choose_action(q_tables[player], state, legal_moves, epsilon, rng)


def _update_q(
    q_table: QTable,
    state: ObservationKey,
    action: ActionKey,
    reward: float,
    next_state: ObservationKey | None,
    next_legal_moves: List[Move],
    alpha: float,
    gamma: float,
) -> None:
    state_values = q_table.setdefault(state, {})
    old_value = state_values.get(action, 0.0)
    bootstrap = 0.0
    if next_state is not None:
        bootstrap = _max_q(q_table, next_state, next_legal_moves)
    state_values[action] = old_value + alpha * (reward + gamma * bootstrap - old_value)


def _count_unknown(observation: Tuple[Tuple[Piece, ...], ...]) -> int:
    return sum(piece == Piece.UNKNOWN for row in observation for piece in row)


def save_q_tables(q_tables: MutableMapping[int, QTable], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        str(player): {
            ",".join(str(value) for value in state): actions
            for state, actions in q_table.items()
        }
        for player, q_table in q_tables.items()
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_q_tables(path: Path) -> MutableMapping[int, QTable]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(player): {
            tuple(int(value) for value in state.split(",")): {
                action: float(score) for action, score in actions.items()
            }
            for state, actions in q_table.items()
        }
        for player, q_table in payload.items()
    }


def _result_rates(winner_counts: Dict[int, int], total_games: int) -> Dict[str, float]:
    total = max(total_games, 1)
    return {
        "draw_rate": winner_counts[0] / total,
        "white_win_rate": winner_counts[1] / total,
        "black_win_rate": winner_counts[2] / total,
    }


def _decisive_win_rate(win_rate: float, loss_rate: float) -> float:
    decisive_rate = win_rate + loss_rate
    if decisive_rate == 0:
        return 0.0
    return win_rate / decisive_rate


def _evaluate_vs_random(
    q_tables: MutableMapping[int, QTable],
    learned_player: int,
    games: int,
    seed: int,
) -> Dict[str, float]:
    rng = random.Random(seed)
    winner_counts = {0: 0, 1: 0, 2: 0}

    for _ in range(games):
        game = Minichess()
        memories: Dict[int, MemoryState | None] = {1: None, 2: None}
        while not game.game_over:
            player = game.current_player
            legal_moves = game.get_all_valid_moves()
            if not legal_moves:
                break

            observation = game.get_observation(player)
            memories[player] = update_memory(memories[player], observation)
            if player == learned_player:
                state = observation_key(memories[player])  # type: ignore[arg-type]
                move = _choose_action(q_tables[player], state, legal_moves, 0.0, rng)
            else:
                move = rng.choice(legal_moves)

            game.step(move, fog=True)

        winner = game.winner if game.winner is not None else 0
        winner_counts[winner] += 1

    rates = _result_rates(winner_counts, games)
    if learned_player == 1:
        win_rate = rates["white_win_rate"]
        loss_rate = rates["black_win_rate"]
    else:
        win_rate = rates["black_win_rate"]
        loss_rate = rates["white_win_rate"]

    return {
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": rates["draw_rate"],
        "decisive_win_rate": _decisive_win_rate(win_rate, loss_rate),
    }


def train_self_play(
    episodes: int,
    log_file: Path,
    model_file: Path | None = None,
    seed: int = 0,
    alpha: float = 0.2,
    min_alpha: float = 0.05,
    gamma: float = 0.95,
    epsilon_start: float = 0.4,
    epsilon_end: float = 0.05,
    report_every: int = 10,
    eval_games: int = 20,
) -> MutableMapping[int, QTable]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger(__name__)
    rng = random.Random(seed)
    q_tables: MutableMapping[int, QTable] = {1: {}, 2: {}}

    probe = Minichess()
    white_probe = probe.reset(fog=True)
    logger.info(
        "observation_check | player=White | unknown_squares=%s | cells=%s",
        _count_unknown(white_probe),
        len(observation_key(white_probe)),
    )
    logger.info(
        "training_start | episodes=%s | seed=%s | learning_rate=%.3f->%.3f | discount=%.3f | exploration=%.3f->%.3f | evaluation_games=%s",
        episodes,
        seed,
        alpha,
        min_alpha,
        gamma,
        epsilon_start,
        epsilon_end,
        eval_games,
    )

    winner_counts = {0: 0, 1: 0, 2: 0}
    recent_lengths: List[int] = []
    best_score = -1.0
    best_episode = 0

    for episode in range(1, episodes + 1):
        decay = (episode - 1) / max(episodes - 1, 1)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * decay
        current_alpha = alpha + (min_alpha - alpha) * decay
        game = Minichess()
        pending: Dict[int, List[object]] = {}
        memories: Dict[int, MemoryState | None] = {1: None, 2: None}

        while not game.game_over:
            player = game.current_player
            observation = game.get_observation(player)
            memories[player] = update_memory(memories[player], observation)
            state = observation_key(memories[player])  # type: ignore[arg-type]
            legal_moves = game.get_all_valid_moves()

            if player in pending:
                prev_state, prev_action, accumulated_reward = pending.pop(player)
                _update_q(
                    q_tables[player],
                    prev_state,  # type: ignore[arg-type]
                    prev_action,  # type: ignore[arg-type]
                    accumulated_reward,  # type: ignore[arg-type]
                    state,
                    legal_moves,
                    current_alpha,
                    gamma,
                )

            if not legal_moves:
                break

            move = _choose_action(q_tables[player], state, legal_moves, epsilon, rng)
            _, reward, done, _ = game.step(move, fog=True)
            opponent = 3 - player

            if opponent in pending:
                pending[opponent][2] = float(pending[opponent][2]) - reward

            if done:
                _update_q(
                    q_tables[player],
                    state,
                    action_key(move),
                    reward,
                    None,
                    [],
                    current_alpha,
                    gamma,
                )
                for pending_player, (
                    prev_state,
                    prev_action,
                    accumulated_reward,
                ) in list(pending.items()):
                    _update_q(
                        q_tables[pending_player],
                        prev_state,  # type: ignore[arg-type]
                        prev_action,  # type: ignore[arg-type]
                        accumulated_reward,  # type: ignore[arg-type]
                        None,
                        [],
                        current_alpha,
                        gamma,
                    )
                break

            pending[player] = [state, action_key(move), reward]

        winner = game.winner if game.winner is not None else 0
        winner_counts[winner] += 1
        recent_lengths.append(game.move_count)

        if episode % report_every == 0 or episode == episodes:
            window = len(recent_lengths)
            avg_len = sum(recent_lengths) / max(window, 1)
            rates = _result_rates(winner_counts, episode)
            white_eval = _evaluate_vs_random(q_tables, 1, eval_games, seed + episode * 2)
            black_eval = _evaluate_vs_random(
                q_tables, 2, eval_games, seed + episode * 2 + 1
            )
            eval_score = (
                white_eval["decisive_win_rate"] + black_eval["decisive_win_rate"]
            ) / 2
            if eval_score > best_score:
                best_score = eval_score
                best_episode = episode
                if model_file is not None:
                    save_q_tables(q_tables, model_file)
            logger.info(
                "\n"
                "episode=%s | exploration=%.3f | learning_rate=%.3f | average_length=%.1f\n"
                "self_play white_win / black_win / draw = %.1f / %.1f / %.1f\n"
                "white_eval_vs_random decisive_win / win - loss - draw = %.1f | %.1f - %.1f - %.1f\n"
                "black_eval_vs_random decisive_win / win - loss - draw = %.1f | %.1f - %.1f - %.1f\n"
                "best_score=%s at_episode=%s | q_states white/black=%s/%s\n"
                "======",
                episode,
                epsilon,
                current_alpha,
                avg_len,
                rates["white_win_rate"] * 100,
                rates["black_win_rate"] * 100,
                rates["draw_rate"] * 100,
                white_eval["decisive_win_rate"] * 100,
                white_eval["win_rate"] * 100,
                white_eval["loss_rate"] * 100,
                white_eval["draw_rate"] * 100,
                black_eval["decisive_win_rate"] * 100,
                black_eval["win_rate"] * 100,
                black_eval["loss_rate"] * 100,
                black_eval["draw_rate"] * 100,
                f"{best_score * 100:.1f}",
                best_episode,
                len(q_tables[1]),
                len(q_tables[2]),
            )
            recent_lengths.clear()

    rates = _result_rates(winner_counts, episodes)
    white_eval = _evaluate_vs_random(q_tables, 1, eval_games, seed + episodes * 2)
    black_eval = _evaluate_vs_random(q_tables, 2, eval_games, seed + episodes * 2 + 1)
    final_score = (
        white_eval["decisive_win_rate"] + black_eval["decisive_win_rate"]
    ) / 2
    if model_file is not None and best_episode == 0:
        save_q_tables(q_tables, model_file)

    logger.info(
        "\n"
        "training_done\n"
        "self_play white_win / black_win / draw = %.1f / %.1f / %.1f\n"
        "white_eval_vs_random decisive_win / win - loss - draw = %.1f | %.1f - %.1f - %.1f\n"
        "black_eval_vs_random decisive_win / win - loss - draw = %.1f | %.1f - %.1f - %.1f\n"
        "final_score=%.1f | best_score=%.1f at_episode=%s\n"
        "model_file=%s\n"
        "log_file=%s\n"
        "======",
        rates["white_win_rate"] * 100,
        rates["black_win_rate"] * 100,
        rates["draw_rate"] * 100,
        white_eval["decisive_win_rate"] * 100,
        white_eval["win_rate"] * 100,
        white_eval["loss_rate"] * 100,
        white_eval["draw_rate"] * 100,
        black_eval["decisive_win_rate"] * 100,
        black_eval["win_rate"] * 100,
        black_eval["loss_rate"] * 100,
        black_eval["draw_rate"] * 100,
        final_score * 100,
        best_score * 100,
        best_episode,
        model_file,
        log_file,
    )
    return q_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fog-observation Q-learning self-play experiment."
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", type=Path, default=Path("logs/fog_q_learning.log"))
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--min-alpha", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=0.4)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    args = parser.parse_args()

    train_self_play(
        episodes=args.episodes,
        log_file=args.log_file,
        model_file=args.model_file,
        seed=args.seed,
        alpha=args.alpha,
        min_alpha=args.min_alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        report_every=args.report_every,
        eval_games=args.eval_games,
    )


if __name__ == "__main__":
    main()
