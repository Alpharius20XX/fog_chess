from __future__ import annotations

import random
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fog_chess.chess import Minichess, Piece
from fog_chess.human import parse_move_input
from fog_chess.rl import QTable, choose_action_for_game, load_q_tables, update_memory

DEFAULT_MODEL_PATH = Path("models/fog_q_learning.json")

PIECE_ICONS = {
    Piece.WHITE_KING: "♔",
    Piece.WHITE_QUEEN: "♕",
    Piece.WHITE_ROOK: "♖",
    Piece.WHITE_BISHOP: "♗",
    Piece.WHITE_KNIGHT: "♘",
    Piece.WHITE_PAWN: "♙",
    Piece.BLACK_KING: "♚",
    Piece.BLACK_QUEEN: "♛",
    Piece.BLACK_ROOK: "♜",
    Piece.BLACK_BISHOP: "♝",
    Piece.BLACK_KNIGHT: "♞",
    Piece.BLACK_PAWN: "♟",
    Piece.UNKNOWN: "?",
    Piece.EMPTY: "",
}

PROMOTION_NAMES = {
    "Queen": ("Q", "q"),
    "Rook": ("R", "r"),
    "Bishop": ("B", "b"),
    "Knight": ("N", "n"),
}


def _empty_q_tables() -> dict[int, QTable]:
    return {1: {}, 2: {}}


@st.cache_resource
def _load_model(model_path: str) -> dict[int, QTable]:
    path = Path(model_path)
    if not path.exists():
        return _empty_q_tables()
    q_tables = load_q_tables(path)
    return {1: q_tables.get(1, {}), 2: q_tables.get(2, {})}


def _player_name(player: int) -> str:
    return "White" if player == 1 else "Black"


def _square_name(square: tuple[int, int]) -> str:
    row, col = square
    return f"{'abcde'[col]}{5 - row}"


def _is_human_piece(piece: Piece, human_player: int) -> bool:
    return piece.is_white() if human_player == 1 else piece.is_black()


def _changed_squares(
    before: tuple[tuple[Piece, ...], ...], after: tuple[tuple[Piece, ...], ...]
) -> set[tuple[int, int]]:
    changed = set()
    for row in range(5):
        for col in range(5):
            if before[row][col] != after[row][col]:
                changed.add((row, col))
    return changed


def _piece_css_class(piece: Piece) -> str:
    if piece.is_white():
        return "white-piece"
    if piece.is_black():
        return "black-piece"
    if piece == Piece.UNKNOWN:
        return "unknown-piece"
    return "empty-square"


def _captured_icons(pieces: list[Piece]) -> str:
    if not pieces:
        return "None"
    return " ".join(PIECE_ICONS[piece] for piece in pieces)


def _ensure_state() -> None:
    if "game" not in st.session_state:
        st.session_state.game = Minichess()
    if "human_player" not in st.session_state:
        st.session_state.human_player = 1
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rng" not in st.session_state:
        st.session_state.rng = random.Random(0)
    if "selected_square" not in st.session_state:
        st.session_state.selected_square = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "board_click_counter" not in st.session_state:
        st.session_state.board_click_counter = 0
    if "captured_by_human" not in st.session_state:
        st.session_state.captured_by_human = []
    if "captured_by_ai" not in st.session_state:
        st.session_state.captured_by_ai = []
    if "recent_moves" not in st.session_state:
        st.session_state.recent_moves = []
    if "changed_visibility" not in st.session_state:
        st.session_state.changed_visibility = set()
    if "ai_memory" not in st.session_state:
        st.session_state.ai_memory = None


def _visible_ai_move_info(
    move,
    before_observation: tuple[tuple[Piece, ...], ...],
    after_observation: tuple[tuple[Piece, ...], ...],
):
    start_piece = before_observation[move.start[0]][move.start[1]]
    end_piece = after_observation[move.end[0]][move.end[1]]
    start_visible = start_piece != Piece.UNKNOWN
    end_visible = end_piece != Piece.UNKNOWN

    if start_visible and end_visible:
        return ("ai", move.start, move.end, end_piece)
    if end_visible:
        return ("ai", move.end, move.end, end_piece)
    return None


def _ai_move(
    q_tables: dict[int, QTable],
    before_human_observation: tuple[tuple[Piece, ...], ...] | None = None,
) -> None:
    game: Minichess = st.session_state.game
    if game.game_over:
        return

    human_player = st.session_state.human_player
    if before_human_observation is None:
        before_human_observation = game.get_observation(human_player)

    legal_moves = game.get_all_valid_moves()
    if not legal_moves:
        return

    ai_player = game.current_player
    st.session_state.ai_memory = update_memory(
        st.session_state.ai_memory, game.get_observation(ai_player)
    )
    move = choose_action_for_game(
        q_tables,
        game,
        st.session_state.rng,
        epsilon=0.0,
        memory=st.session_state.ai_memory,
    )
    game.step(move, fog=True)
    if move.captured:
        st.session_state.captured_by_ai.append(move.captured)
    after_human_observation = game.get_observation(human_player)
    visible_move_info = _visible_ai_move_info(
        move, before_human_observation, after_human_observation
    )
    if visible_move_info is not None:
        st.session_state.recent_moves.append(visible_move_info)
    st.session_state.messages.append(
        f"AI ({_player_name(3 - human_player)}) moved."
    )


def _push_undo_state() -> None:
    st.session_state.history.append(
        (
            st.session_state.game.copy(),
            list(st.session_state.messages),
            st.session_state.selected_square,
            list(st.session_state.captured_by_human),
            list(st.session_state.captured_by_ai),
            list(st.session_state.recent_moves),
            set(st.session_state.changed_visibility),
            st.session_state.ai_memory,
        )
    )


def _undo_last_turn() -> None:
    if not st.session_state.history:
        return
    (
        game,
        messages,
        selected_square,
        captured_by_human,
        captured_by_ai,
        recent_moves,
        changed_visibility,
        ai_memory,
    ) = st.session_state.history.pop()
    st.session_state.game = game
    st.session_state.messages = messages
    st.session_state.selected_square = selected_square
    st.session_state.captured_by_human = captured_by_human
    st.session_state.captured_by_ai = captured_by_ai
    st.session_state.recent_moves = recent_moves
    st.session_state.changed_visibility = changed_visibility
    st.session_state.ai_memory = ai_memory


def _start_new_game(human_player: int, q_tables: dict[int, QTable]) -> None:
    st.session_state.game = Minichess()
    st.session_state.human_player = human_player
    st.session_state.messages = [f"New game started. You are {_player_name(human_player)}."]
    st.session_state.selected_square = None
    st.session_state.history = []
    st.session_state.captured_by_human = []
    st.session_state.captured_by_ai = []
    st.session_state.recent_moves = []
    st.session_state.changed_visibility = set()
    st.session_state.ai_memory = None
    if st.session_state.game.current_player != human_player:
        _ai_move(q_tables)


def _choose_promotion_move(candidates: list, promotion_choice: str):
    if len(candidates) == 1:
        return candidates[0]
    preferred_symbols = PROMOTION_NAMES[promotion_choice]
    for move in candidates:
        if move.promotion and move.promotion.to_string() in preferred_symbols:
            return move
    return candidates[0]


def _play_human_move(move, q_tables: dict[int, QTable]) -> None:
    game: Minichess = st.session_state.game
    human_player = st.session_state.human_player
    before_observation = game.get_observation(human_player)
    _push_undo_state()
    game.step(move, fog=True)
    if move.captured:
        st.session_state.captured_by_human.append(move.captured)
    st.session_state.recent_moves = [
        ("human", move.start, move.end, move.promotion or move.piece)
    ]
    st.session_state.selected_square = None
    st.session_state.messages.append(f"You played {move.to_notation()}.")
    after_human_observation = game.get_observation(human_player)
    if not game.game_over:
        _ai_move(q_tables, before_human_observation=after_human_observation)
    after_observation = game.get_observation(human_player)
    st.session_state.changed_visibility = _changed_squares(
        before_observation, after_observation
    )


def _handle_square_click(
    row: int,
    col: int,
    q_tables: dict[int, QTable],
    promotion_choice: str,
) -> None:
    game: Minichess = st.session_state.game
    human_player = st.session_state.human_player
    if game.game_over or game.current_player != human_player:
        return

    clicked = (row, col)
    observation = game.get_observation(human_player)
    piece = observation[row][col]
    selected = st.session_state.selected_square

    if selected is None:
        if _is_human_piece(piece, human_player):
            st.session_state.selected_square = clicked
            st.session_state.messages.append(f"Selected {_square_name(clicked)}.")
        return

    if selected == clicked:
        st.session_state.selected_square = None
        return

    valid_moves = game.get_all_valid_moves()
    candidates = [
        move for move in valid_moves if move.start == selected and move.end == clicked
    ]
    if candidates:
        move = _choose_promotion_move(candidates, promotion_choice)
        _play_human_move(move, q_tables)
        return

    if _is_human_piece(piece, human_player):
        st.session_state.selected_square = clicked
        st.session_state.messages.append(f"Selected {_square_name(clicked)}.")
    else:
        st.session_state.messages.append(
            f"No legal move from {_square_name(selected)} to {_square_name(clicked)}."
        )


def _render_clickable_board(
    board: tuple[tuple[Piece, ...], ...],
    q_tables: dict[int, QTable],
    promotion_choice: str,
) -> None:
    st.caption("Click one of your pieces, then click a destination square.")
    selected = st.session_state.selected_square
    recent_move_squares = {
        square
        for move_info in st.session_state.recent_moves[-2:]
        for square in (move_info[1], move_info[2])
    }
    changed_visibility = st.session_state.changed_visibility

    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] button {
            min-height: 62px;
            font-size: 32px;
            line-height: 1;
            border-radius: 4px;
            border: 2px solid #6b4f36;
            background-color: #d9b98c;
            color: #111111;
        }
        div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] button:hover {
            border-color: #2f80ed;
            color: #111111;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    header = st.columns(6)
    header[0].markdown(" ")
    for col_idx, name in enumerate("abcde", start=1):
        header[col_idx].markdown(f"**{name}**")

    for row_idx, row in enumerate(board):
        rank = 5 - row_idx
        cols = st.columns(6)
        cols[0].markdown(f"**{rank}**")
        for col_idx, piece in enumerate(row):
            square = (row_idx, col_idx)
            label = PIECE_ICONS[piece] or " "
            if selected == square:
                label = f"● {label}"
            elif square in recent_move_squares:
                label = f"★ {label}"
            elif square in changed_visibility:
                label = f"◇ {label}"
            if cols[col_idx + 1].button(
                label,
                key=f"square-{row_idx}-{col_idx}",
                help=_square_name(square),
                use_container_width=True,
            ):
                _handle_square_click(row_idx, col_idx, q_tables, promotion_choice)
                st.rerun()


def _render_one_move_animation(
    board: tuple[tuple[Piece, ...], ...], move_info
) -> str:
    if len(move_info) == 4:
        actor, start, end, moved_piece = move_info
    else:
        actor, start, end = move_info
        moved_piece = board[end[0]][end[1]]
    icon = PIECE_ICONS.get(moved_piece, "")
    start_x = start[1] * 60
    start_y = start[0] * 60
    end_x = end[1] * 60
    end_y = end[0] * 60
    delay = "0s" if actor == "human" else "1.45s"
    label = "Your move" if actor == "human" else "AI move"
    return (
        '<div class="animation-piece" style="'
        f"--start-x: {start_x}px; --start-y: {start_y}px; "
        f"--end-x: {end_x}px; --end-y: {end_y}px; "
        f"--move-delay: {delay};"
        f'" title="{label}">{icon}</div>'
    )


def _render_last_move_animation(board: tuple[tuple[Piece, ...], ...]) -> None:
    if not st.session_state.recent_moves:
        return

    rows = [
        """
        <style>
        body { margin: 0; }
        .animation-title {
            font-family: sans-serif;
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .animation-wrap { position: relative; width: 300px; height: 300px; }
        .animation-board {
            border-collapse: collapse;
            position: absolute;
            left: 0;
            top: 0;
        }
        .animation-board td {
            width: 60px;
            height: 60px;
            text-align: center;
            vertical-align: middle;
            font-size: 32px;
            border: 1px solid #5f4630;
            font-family: "Apple Symbols", "DejaVu Sans", "Segoe UI Symbol", serif;
        }
        .animation-board .light { background: #f0d9b5; }
        .animation-board .dark { background: #b58863; }
        .animation-piece {
            position: absolute;
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 38px;
            z-index: 10;
            pointer-events: none;
            font-family: "Apple Symbols", "DejaVu Sans", "Segoe UI Symbol", serif;
            animation: slideMove 1.35s ease-in-out 1;
            animation-delay: var(--move-delay);
            animation-fill-mode: both;
            transform: translate(var(--end-x), var(--end-y));
        }
        @keyframes slideMove {
            from { transform: translate(var(--start-x), var(--start-y)); }
            to { transform: translate(var(--end-x), var(--end-y)); }
        }
        </style>
        <div class="animation-wrap">
        <table class="animation-board">
        """
    ]
    animated_ends = {move_info[2] for move_info in st.session_state.recent_moves[-2:]}
    for row_idx, row in enumerate(board):
        cells = []
        for col_idx, piece in enumerate(row):
            square_colour = "light" if (row_idx + col_idx) % 2 == 0 else "dark"
            shown_icon = "" if (row_idx, col_idx) in animated_ends else PIECE_ICONS[piece]
            cells.append(f'<td class="{square_colour}">{shown_icon}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</table>")
    for move_info in st.session_state.recent_moves[-2:]:
        rows.append(_render_one_move_animation(board, move_info))
    rows.append("</div>")
    st.caption("Recent move animation")
    components.html("".join(rows), height=350)


def _render_captures() -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**You captured**")
        st.markdown(
            f"<div style='min-height:48px;font-size:32px'>"
            f"{_captured_icons(st.session_state.captured_by_human)}</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("**AI captured**")
        st.markdown(
            f"<div style='min-height:48px;font-size:32px'>"
            f"{_captured_icons(st.session_state.captured_by_ai)}</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="Fog Chess vs RL", page_icon="F", layout="centered")
    _ensure_state()

    st.title("Fog Chess vs RL")

    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("Model file", value=str(DEFAULT_MODEL_PATH))
        q_tables = _load_model(model_path)
        if Path(model_path).exists():
            st.success(f"Loaded model: {model_path}")
        else:
            st.warning("Model file not found. AI will behave like an untrained policy.")

        side = st.radio("Your side", ["White", "Black"], horizontal=True)
        human_player = 1 if side == "White" else 2
        promotion_choice = st.selectbox(
            "Click promotion piece", ["Queen", "Rook", "Bishop", "Knight"]
        )
        if st.button("New game", type="primary"):
            _start_new_game(human_player, q_tables)
        if st.button("Undo last turn", disabled=not st.session_state.history):
            _undo_last_turn()
            st.rerun()

        st.caption("Train a model with `python -m fog_chess.rl --model-file models/fog_q_learning.json`.")

    game: Minichess = st.session_state.game
    human_player = st.session_state.human_player

    st.subheader(f"Your fog view ({_player_name(human_player)})")
    observation = game.get_observation(human_player)
    board_col, animation_col = st.columns([1.25, 1])
    with board_col:
        _render_clickable_board(observation, q_tables, promotion_choice)
    with animation_col:
        _render_last_move_animation(observation)
    _render_captures()

    status = "Game over" if game.game_over else f"{_player_name(game.current_player)} to move"
    st.write(f"Move count: {game.move_count} | {status}")

    if game.game_over:
        if game.winner == 0:
            st.info("Draw.")
        elif game.winner == human_player:
            st.success("You won.")
        else:
            st.error("AI won.")
    elif game.current_player == human_player:
        with st.form("move_form", clear_on_submit=True):
            move_text = st.text_input("Your move", placeholder="e.g. a2a3 or a2a1=Q")
            submitted = st.form_submit_button("Play move")

        if submitted:
            valid_moves = game.get_all_valid_moves()
            move = parse_move_input(move_text, valid_moves)
            if move is None:
                st.error("Invalid move. Try notation like a2a3 or a2a1=Q.")
            else:
                _play_human_move(move, q_tables)
                st.rerun()
    else:
        if st.button("Let AI move"):
            _ai_move(q_tables)
            st.rerun()

    with st.expander("Move log", expanded=True):
        for message in reversed(st.session_state.messages[-12:]):
            st.write(message)


if __name__ == "__main__":
    main()
