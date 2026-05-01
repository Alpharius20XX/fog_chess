from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np


class Piece(Enum):
    UNKNOWN = 0
    EMPTY = 1
    WHITE_PAWN = 2
    WHITE_KNIGHT = 3
    WHITE_BISHOP = 4
    WHITE_ROOK = 5
    WHITE_QUEEN = 6
    WHITE_KING = 7
    BLACK_PAWN = 8
    BLACK_KNIGHT = 9
    BLACK_BISHOP = 10
    BLACK_ROOK = 11
    BLACK_QUEEN = 12
    BLACK_KING = 13

    def is_white(self) -> bool:
        return self in (
            Piece.WHITE_PAWN,
            Piece.WHITE_KNIGHT,
            Piece.WHITE_BISHOP,
            Piece.WHITE_ROOK,
            Piece.WHITE_QUEEN,
            Piece.WHITE_KING,
        )

    def is_black(self) -> bool:
        return self in (
            Piece.BLACK_PAWN,
            Piece.BLACK_KNIGHT,
            Piece.BLACK_BISHOP,
            Piece.BLACK_ROOK,
            Piece.BLACK_QUEEN,
            Piece.BLACK_KING,
        )

    def get_value(self, side: bool):
        sign = 1 if side else -1
        match self:
            case Piece.WHITE_PAWN | Piece.BLACK_PAWN:
                return sign * 100
            case Piece.WHITE_KNIGHT | Piece.BLACK_KNIGHT:
                return sign * 320
            case Piece.WHITE_BISHOP | Piece.BLACK_BISHOP:
                return sign * 330
            case Piece.WHITE_ROOK | Piece.BLACK_ROOK:
                return sign * 500
            case Piece.WHITE_QUEEN | Piece.BLACK_QUEEN:
                return sign * 900
            case Piece.WHITE_KING | Piece.BLACK_KING:
                return sign * 20000
            case _:
                return 0

    def to_string(self):
        match self:
            case Piece.UNKNOWN:
                return "?"
            case Piece.EMPTY:
                return " "
            case Piece.WHITE_PAWN:
                return "P"
            case Piece.WHITE_KNIGHT:
                return "N"
            case Piece.WHITE_BISHOP:
                return "B"
            case Piece.WHITE_ROOK:
                return "R"
            case Piece.WHITE_QUEEN:
                return "Q"
            case Piece.WHITE_KING:
                return "K"
            case Piece.BLACK_PAWN:
                return "p"
            case Piece.BLACK_KNIGHT:
                return "n"
            case Piece.BLACK_BISHOP:
                return "b"
            case Piece.BLACK_ROOK:
                return "r"
            case Piece.BLACK_QUEEN:
                return "q"
            case Piece.BLACK_KING:
                return "k"
            case _:
                return ""


@dataclass(frozen=True)
class Move:
    start: Tuple[int, int]
    end: Tuple[int, int]
    piece: Piece
    captured: Optional[Piece] = None
    promotion: Optional[Piece] = None
    is_check: bool = False
    is_checkmate: bool = False
    is_en_passant: bool = False
    is_castling: bool = False

    def to_notation(self, board_size: int = 5) -> str:
        cols = "abcdefgh"[:board_size]
        s = f"{cols[self.start[1]]}{board_size - self.start[0]}"
        e = f"{cols[self.end[1]]}{board_size - self.end[0]}"
        notation = f"{s}{e}"
        if self.promotion:
            notation += f"={self.promotion.to_string().upper()}"
        if self.is_checkmate:
            notation += "#"
        elif self.is_check:
            notation += "+"
        return notation


_SLIDING_PIECES = frozenset(
    {
        Piece.WHITE_BISHOP,
        Piece.BLACK_BISHOP,
        Piece.WHITE_ROOK,
        Piece.BLACK_ROOK,
        Piece.WHITE_QUEEN,
        Piece.BLACK_QUEEN,
        Piece.WHITE_KING,
        Piece.BLACK_KING,
    }
)

_WHITE_PROMOTIONS = [
    Piece.WHITE_QUEEN,
    Piece.WHITE_ROOK,
    Piece.WHITE_BISHOP,
    Piece.WHITE_KNIGHT,
]
_BLACK_PROMOTIONS = [
    Piece.BLACK_QUEEN,
    Piece.BLACK_ROOK,
    Piece.BLACK_BISHOP,
    Piece.BLACK_KNIGHT,
]


class Minichess:
    def __init__(self, board_size: int = 5, max_moves: int = 100, points_tiebreak: bool = False):
        self.board_size = board_size
        self.max_moves = max_moves
        self.points_tiebreak = points_tiebreak
        self.reset()

    def reset(self):
        self.board = np.array(
            [
                [
                    Piece.BLACK_KING,
                    Piece.BLACK_QUEEN,
                    Piece.BLACK_BISHOP,
                    Piece.BLACK_KNIGHT,
                    Piece.BLACK_ROOK,
                ],
                [
                    Piece.BLACK_PAWN,
                    Piece.BLACK_PAWN,
                    Piece.BLACK_PAWN,
                    Piece.BLACK_PAWN,
                    Piece.BLACK_PAWN,
                ],
                [Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY],
                [
                    Piece.WHITE_PAWN,
                    Piece.WHITE_PAWN,
                    Piece.WHITE_PAWN,
                    Piece.WHITE_PAWN,
                    Piece.WHITE_PAWN,
                ],
                [
                    Piece.WHITE_KING,
                    Piece.WHITE_QUEEN,
                    Piece.WHITE_BISHOP,
                    Piece.WHITE_KNIGHT,
                    Piece.WHITE_ROOK,
                ],
            ],
            dtype=object,
        )
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history: List[Move] = []
        self.move_count = 0
        self._fog_cache_initialized = False
        self._fog_base_log: dict = {1: [], 2: []}
        self._fog_state_cache: dict = {1: None, 2: None}
        return self.get_state()

    def _make_fresh_game(self) -> "Minichess":
        return Minichess(max_moves=self.max_moves, points_tiebreak=self.points_tiebreak)

    def get_state(self):
        return tuple(tuple(row) for row in self.board)

    def get_fog_state(self, player: int):
        bs = self.board_size
        visible: set = set()
        original_player = self.current_player
        self.current_player = player
        try:
            for row in range(bs):
                for col in range(bs):
                    piece = self.board[row, col]
                    if piece != Piece.EMPTY and self.is_friendly(piece, player):
                        visible.add((row, col))
                        for move in self.get_piece_moves(row, col, check_legal=False):
                            visible.add(move.end)
        finally:
            self.current_player = original_player

        fog_board = np.empty((bs, bs), dtype=object)
        for row in range(bs):
            for col in range(bs):
                fog_board[row, col] = self.board[row, col] if (row, col) in visible else Piece.UNKNOWN
        return tuple(tuple(r) for r in fog_board)

    def get_fog_log(self, player: int) -> List[Tuple[int, int, int]]:
        bs = self.board_size
        self._ensure_fog_cache()
        fog_state = self._fog_state_cache[player]
        turn = self.move_count
        unknowns = [
            (turn, row * bs + col, 0)
            for row in range(bs)
            for col in range(bs)
            if fog_state[row][col] == Piece.UNKNOWN
        ]
        return self._fog_base_log[player] + unknowns

    def get_log(self) -> List[Tuple[int, int, int]]:
        bs = self.board_size
        replay = self._make_fresh_game()
        states = [replay.get_state()]

        for move in self.move_history:
            replay._make_move_internal(move)
            replay.current_player = 3 - replay.current_player
            states.append(replay.get_state())

        log = []
        for turn, state in enumerate(states):
            prev = states[turn - 1] if turn > 0 else None
            for row in range(bs):
                for col in range(bs):
                    sq = row * bs + col
                    piece = state[row][col]
                    if turn == 0:
                        log.append((turn, sq, piece.value))
                    elif prev is not None and piece != prev[row][col]:
                        log.append((turn, sq, piece.value))

        return log

    def get_legal_move_indices(self, player: int) -> List[Tuple[int, int]]:
        bs = self.board_size
        original_player = self.current_player
        self.current_player = player
        try:
            moves = self.get_all_valid_moves()
        finally:
            self.current_player = original_player
        return [(m.start[0] * bs + m.start[1], m.end[0] * bs + m.end[1]) for m in moves]

    def get_legal_moves_and_indices(
        self, player: int
    ) -> Tuple[List["Move"], List[Tuple[int, int]]]:
        """Return (moves, action_indices) together to avoid computing valid moves twice."""
        bs = self.board_size
        original_player = self.current_player
        self.current_player = player
        try:
            moves = self.get_all_valid_moves()
        finally:
            self.current_player = original_player
        indices = [(m.start[0] * bs + m.start[1], m.end[0] * bs + m.end[1]) for m in moves]
        return moves, indices

    def _ensure_fog_cache(self):
        if self._fog_cache_initialized:
            return
        bs = self.board_size
        replay = self._make_fresh_game()
        fog_states = {p: [replay.get_fog_state(p)] for p in (1, 2)}
        for move in self.move_history:
            replay._make_move_internal(move)
            replay.current_player = 3 - replay.current_player
            for p in (1, 2):
                fog_states[p].append(replay.get_fog_state(p))
        for p in (1, 2):
            states = fog_states[p]
            base_log = []
            for turn, state in enumerate(states):
                prev = states[turn - 1] if turn > 0 else None
                for row in range(bs):
                    for col in range(bs):
                        sq = row * bs + col
                        piece = state[row][col]
                        if turn == 0:
                            if piece != Piece.UNKNOWN:
                                base_log.append((turn, sq, piece.value))
                        else:
                            prev_piece = prev[row][col]  # type: ignore[index]
                            if piece != prev_piece and piece != Piece.UNKNOWN:
                                base_log.append((turn, sq, piece.value))
            self._fog_base_log[p] = base_log
            self._fog_state_cache[p] = states[-1]
        self._fog_cache_initialized = True

    def copy(self):
        new_game = object.__new__(type(self))
        new_game.board_size = self.board_size
        new_game.max_moves = self.max_moves
        new_game.points_tiebreak = self.points_tiebreak
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.move_history = self.move_history.copy()
        new_game.move_count = self.move_count
        new_game._fog_cache_initialized = False
        new_game._fog_base_log = {1: [], 2: []}
        new_game._fog_state_cache = {1: None, 2: None}
        return new_game

    def sq_to_rc(self, sq: int) -> Tuple[int, int]:
        return sq // self.board_size, sq % self.board_size

    @staticmethod
    def is_white_piece(piece: Piece) -> bool:
        return piece.is_white()

    @staticmethod
    def is_black_piece(piece: Piece) -> bool:
        return piece.is_black()

    def is_enemy(self, piece: Piece, player: int) -> bool:
        return piece.is_black() if player == 1 else piece.is_white()

    def is_friendly(self, piece: Piece, player: int) -> bool:
        return piece.is_white() if player == 1 else piece.is_black()

    def get_piece_moves(
        self, row: int, col: int, check_legal: bool = True
    ) -> List[Move]:
        piece = self.board[row, col]
        if piece == Piece.EMPTY or not self.is_friendly(piece, self.current_player):
            return []

        match piece:
            case Piece.WHITE_PAWN | Piece.BLACK_PAWN:
                moves = self._get_pawn_moves(row, col)
            case Piece.WHITE_KNIGHT | Piece.BLACK_KNIGHT:
                moves = self._get_knight_moves(row, col)
            case Piece.WHITE_BISHOP | Piece.BLACK_BISHOP:
                moves = self._get_bishop_moves(row, col)
            case Piece.WHITE_ROOK | Piece.BLACK_ROOK:
                moves = self._get_rook_moves(row, col)
            case Piece.WHITE_QUEEN | Piece.BLACK_QUEEN:
                moves = self._get_queen_moves(row, col)
            case Piece.WHITE_KING | Piece.BLACK_KING:
                moves = self._get_king_moves(row, col)
            case _:
                moves = []

        if not check_legal:
            return moves

        legal_moves = []
        for move in moves:
            test_game = self.copy()
            test_game._make_move_internal(move)
            if not test_game._is_in_check(self.current_player):
                legal_moves.append(move)
        return legal_moves

    def _get_pawn_moves(self, row: int, col: int) -> List[Move]:
        moves = []
        piece = self.board[row, col]
        bs = self.board_size

        if self.current_player == 1:
            direction = -1
            promotion_row = 0
            promotions = _WHITE_PROMOTIONS
        else:
            direction = 1
            promotion_row = bs - 1
            promotions = _BLACK_PROMOTIONS

        new_row = row + direction

        if 0 <= new_row < bs and self.board[new_row, col] == Piece.EMPTY:
            if new_row == promotion_row:
                for promo in promotions:
                    moves.append(
                        Move((row, col), (new_row, col), piece, promotion=promo)
                    )
            else:
                moves.append(Move((row, col), (new_row, col), piece))

        for dc in [-1, 1]:
            new_col = col + dc
            if 0 <= new_row < bs and 0 <= new_col < bs:
                target = self.board[new_row, new_col]
                if target != Piece.EMPTY and self.is_enemy(target, self.current_player):
                    if new_row == promotion_row:
                        for promo in promotions:
                            moves.append(
                                Move(
                                    (row, col),
                                    (new_row, new_col),
                                    piece,
                                    captured=target,
                                    promotion=promo,
                                )
                            )
                    else:
                        moves.append(
                            Move((row, col), (new_row, new_col), piece, captured=target)
                        )
        return moves

    def _get_knight_moves(self, row: int, col: int) -> List[Move]:
        moves = []
        piece = self.board[row, col]
        bs = self.board_size
        for dr, dc in [
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < bs and 0 <= nc < bs:
                target = self.board[nr, nc]
                if target == Piece.EMPTY or self.is_enemy(target, self.current_player):
                    moves.append(
                        Move(
                            (row, col),
                            (nr, nc),
                            piece,
                            captured=None if target == Piece.EMPTY else target,
                        )
                    )
        return moves

    def _get_sliding_moves(
        self, row: int, col: int, directions: List[Tuple[int, int]]
    ) -> List[Move]:
        moves = []
        piece = self.board[row, col]
        bs = self.board_size
        for dr, dc in directions:
            for i in range(1, bs):
                nr, nc = row + dr * i, col + dc * i
                if not (0 <= nr < bs and 0 <= nc < bs):
                    break
                target = self.board[nr, nc]
                if target == Piece.EMPTY:
                    moves.append(Move((row, col), (nr, nc), piece))
                elif self.is_enemy(target, self.current_player):
                    moves.append(Move((row, col), (nr, nc), piece, captured=target))
                    break
                else:
                    break
        return moves

    def _get_bishop_moves(self, row: int, col: int) -> List[Move]:
        return self._get_sliding_moves(row, col, [(-1, -1), (-1, 1), (1, -1), (1, 1)])

    def _get_rook_moves(self, row: int, col: int) -> List[Move]:
        return self._get_sliding_moves(row, col, [(-1, 0), (1, 0), (0, -1), (0, 1)])

    def _get_queen_moves(self, row: int, col: int) -> List[Move]:
        return self._get_sliding_moves(
            row,
            col,
            [
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ],
        )

    def _get_king_moves(self, row: int, col: int) -> List[Move]:
        moves = []
        piece = self.board[row, col]
        bs = self.board_size
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < bs and 0 <= nc < bs:
                    target = self.board[nr, nc]
                    if target == Piece.EMPTY or self.is_enemy(
                        target, self.current_player
                    ):
                        moves.append(
                            Move(
                                (row, col),
                                (nr, nc),
                                piece,
                                captured=None if target == Piece.EMPTY else target,
                            )
                        )
        return moves

    def get_all_valid_moves(self) -> List[Move]:
        moves = []
        bs = self.board_size
        for row in range(bs):
            for col in range(bs):
                piece = self.board[row, col]
                if piece != Piece.EMPTY and self.is_friendly(
                    piece, self.current_player
                ):
                    moves.extend(self.get_piece_moves(row, col))
        return moves

    def _find_king(self, player: int):
        king = Piece.WHITE_KING if player == 1 else Piece.BLACK_KING
        positions = np.argwhere(self.board == king)
        return tuple(positions[0]) if len(positions) else None

    def _is_square_attacked(self, row: int, col: int, by_player: int) -> bool:
        original_player = self.current_player
        self.current_player = by_player
        try:
            bs = self.board_size
            for r in range(bs):
                for c in range(bs):
                    piece = self.board[r, c]
                    if piece != Piece.EMPTY and self.is_friendly(piece, by_player):
                        for move in self.get_piece_moves(r, c, check_legal=False):
                            if move.end == (row, col):
                                return True
            return False
        finally:
            self.current_player = original_player

    def _is_in_check(self, player: int) -> bool:
        king_pos = self._find_king(player)
        if king_pos is None:
            return False
        return self._is_square_attacked(king_pos[0], king_pos[1], 3 - player)

    def _make_move_internal(self, move: Move):
        sr, sc = move.start
        er, ec = move.end
        piece = move.promotion if move.promotion else self.board[sr, sc]
        self.board[er, ec] = piece
        self.board[sr, sc] = Piece.EMPTY

    def _compute_material_winner(self) -> int:
        white = sum(
            abs(self.board[r, c].get_value(True))
            for r in range(self.board_size)
            for c in range(self.board_size)
            if self.board[r, c].is_white()
        )
        black = sum(
            abs(self.board[r, c].get_value(True))
            for r in range(self.board_size)
            for c in range(self.board_size)
            if self.board[r, c].is_black()
        )
        if white > black:
            return 1
        if black > white:
            return 2
        return 0

    def make_move(self, move: Move):
        if self.game_over:
            return self.get_state(), 0, True

        reward = 0
        if move.captured:
            reward = abs(move.captured.get_value(True)) / 100
        if move.promotion:
            reward += 5

        self._make_move_internal(move)
        self.move_history.append(move)
        self.move_count += 1
        self.current_player = 3 - self.current_player

        if self._fog_cache_initialized:
            bs = self.board_size
            turn = self.move_count
            for p in (1, 2):
                prev_fog = self._fog_state_cache[p]
                new_fog = self.get_fog_state(p)
                for row in range(bs):
                    for col in range(bs):
                        piece = new_fog[row][col]
                        if piece != prev_fog[row][col] and piece != Piece.UNKNOWN:
                            self._fog_base_log[p].append((turn, row * bs + col, piece.value))
                self._fog_state_cache[p] = new_fog

        opponent_moves = self.get_all_valid_moves()
        is_check = self._is_in_check(self.current_player)

        if not opponent_moves:
            self.game_over = True
            if is_check:
                self.winner = 3 - self.current_player
                reward = 100
            else:
                self.winner = 0
                reward = 0

        if self.move_count >= self.max_moves:
            self.game_over = True
            if self.points_tiebreak:
                self.winner = self._compute_material_winner()
            else:
                self.winner = 0

        return self.get_state(), reward, self.game_over

    def evaluate_position(self, player: int) -> float:
        if self.winner == player:
            return 100000
        if self.winner == 3 - player:
            return -100000
        if self.winner == 0:
            return 0

        bs = self.board_size
        center_table = np.zeros((bs, bs), dtype=np.int32)
        for r in range(bs):
            for c in range(bs):
                center_table[r, c] = min(r, bs - 1 - r, c, bs - 1 - c) * 5

        score = 0
        for row in range(bs):
            for col in range(bs):
                piece = self.board[row, col]
                if piece == Piece.EMPTY:
                    continue
                multiplier = 1 if self.is_friendly(piece, player) else -1
                score += multiplier * abs(piece.get_value(True))
                if piece in _SLIDING_PIECES:
                    score += multiplier * center_table[row, col] * 2

        original_player = self.current_player
        self.current_player = player
        my_moves = len(self.get_all_valid_moves())
        self.current_player = 3 - player
        opp_moves = len(self.get_all_valid_moves())
        self.current_player = original_player

        score += (my_moves - opp_moves) * 10
        if self._is_in_check(3 - player):
            score += 50
        return float(score)

    def print_board(self):
        bs = self.board_size
        cols = "abcdefgh"[:bs]
        print(f"\n  {' '.join(cols)}")
        for r in range(bs):
            row_str = " ".join(p.to_string() for p in self.board[r])
            print(f"{bs - r} {row_str} {bs - r}")
        print(f"  {' '.join(cols)}\n")


class FullChess(Minichess):
    """Standard 8×8 chess with castling, en passant, and double pawn push."""

    def __init__(self, max_moves: int = 300, points_tiebreak: bool = False):
        self.board_size = 8
        self.max_moves = max_moves
        self.points_tiebreak = points_tiebreak
        self.reset()

    def reset(self):
        self.board = np.array(
            [
                [Piece.BLACK_ROOK, Piece.BLACK_KNIGHT, Piece.BLACK_BISHOP, Piece.BLACK_QUEEN,
                 Piece.BLACK_KING, Piece.BLACK_BISHOP, Piece.BLACK_KNIGHT, Piece.BLACK_ROOK],
                [Piece.BLACK_PAWN] * 8,
                [Piece.EMPTY] * 8,
                [Piece.EMPTY] * 8,
                [Piece.EMPTY] * 8,
                [Piece.EMPTY] * 8,
                [Piece.WHITE_PAWN] * 8,
                [Piece.WHITE_ROOK, Piece.WHITE_KNIGHT, Piece.WHITE_BISHOP, Piece.WHITE_QUEEN,
                 Piece.WHITE_KING, Piece.WHITE_BISHOP, Piece.WHITE_KNIGHT, Piece.WHITE_ROOK],
            ],
            dtype=object,
        )
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history: List[Move] = []
        self.move_count = 0
        self._fog_cache_initialized = False
        self._fog_base_log: dict = {1: [], 2: []}
        self._fog_state_cache: dict = {1: None, 2: None}
        self.en_passant_target: Optional[Tuple[int, int]] = None
        self.castling_rights = {
            1: {"kingside": True, "queenside": True},
            2: {"kingside": True, "queenside": True},
        }
        return self.get_state()

    def _make_fresh_game(self) -> "FullChess":
        return FullChess(max_moves=self.max_moves, points_tiebreak=self.points_tiebreak)

    def copy(self):
        new_game = super().copy()
        new_game.en_passant_target = self.en_passant_target
        new_game.castling_rights = {p: dict(r) for p, r in self.castling_rights.items()}
        return new_game

    def _make_move_internal(self, move: Move):
        sr, sc = move.start
        er, ec = move.end
        original_piece = self.board[sr, sc]
        piece = move.promotion if move.promotion else original_piece

        if move.is_en_passant:
            self.board[sr, ec] = Piece.EMPTY

        if move.is_castling:
            if ec > sc:  # kingside
                rook_col = self.board_size - 1
                rook_end_col = ec - 1
            else:  # queenside
                rook_col = 0
                rook_end_col = ec + 1
            self.board[sr, rook_end_col] = self.board[sr, rook_col]
            self.board[sr, rook_col] = Piece.EMPTY

        self.board[er, ec] = piece
        self.board[sr, sc] = Piece.EMPTY

        # Update en passant target
        if original_piece in (Piece.WHITE_PAWN, Piece.BLACK_PAWN) and abs(er - sr) == 2:
            self.en_passant_target = ((sr + er) // 2, sc)
        else:
            self.en_passant_target = None

        # Update castling rights
        if original_piece == Piece.WHITE_KING:
            self.castling_rights[1] = {"kingside": False, "queenside": False}
        elif original_piece == Piece.BLACK_KING:
            self.castling_rights[2] = {"kingside": False, "queenside": False}
        elif original_piece == Piece.WHITE_ROOK:
            if sc == 0:
                self.castling_rights[1]["queenside"] = False
            elif sc == self.board_size - 1:
                self.castling_rights[1]["kingside"] = False
        elif original_piece == Piece.BLACK_ROOK:
            if sc == 0:
                self.castling_rights[2]["queenside"] = False
            elif sc == self.board_size - 1:
                self.castling_rights[2]["kingside"] = False

    def _get_pawn_moves(self, row: int, col: int) -> List[Move]:
        moves = []
        piece = self.board[row, col]
        bs = self.board_size

        if self.current_player == 1:
            direction = -1
            promotion_row = 0
            start_row = bs - 2
            promotions = _WHITE_PROMOTIONS
        else:
            direction = 1
            promotion_row = bs - 1
            start_row = 1
            promotions = _BLACK_PROMOTIONS

        new_row = row + direction

        if 0 <= new_row < bs and self.board[new_row, col] == Piece.EMPTY:
            if new_row == promotion_row:
                for promo in promotions:
                    moves.append(Move((row, col), (new_row, col), piece, promotion=promo))
            else:
                moves.append(Move((row, col), (new_row, col), piece))
                if row == start_row:
                    double_row = new_row + direction
                    if 0 <= double_row < bs and self.board[double_row, col] == Piece.EMPTY:
                        moves.append(Move((row, col), (double_row, col), piece))

        for dc in [-1, 1]:
            new_col = col + dc
            if 0 <= new_row < bs and 0 <= new_col < bs:
                target = self.board[new_row, new_col]
                if target != Piece.EMPTY and self.is_enemy(target, self.current_player):
                    if new_row == promotion_row:
                        for promo in promotions:
                            moves.append(
                                Move((row, col), (new_row, new_col), piece,
                                     captured=target, promotion=promo)
                            )
                    else:
                        moves.append(
                            Move((row, col), (new_row, new_col), piece, captured=target)
                        )
                elif self.en_passant_target == (new_row, new_col):
                    ep_pawn = self.board[row, new_col]
                    moves.append(
                        Move((row, col), (new_row, new_col), piece,
                             captured=ep_pawn, is_en_passant=True)
                    )
        return moves

    def _get_king_moves(self, row: int, col: int) -> List[Move]:
        moves = super()._get_king_moves(row, col)

        player = self.current_player
        rights = self.castling_rights[player]
        bs = self.board_size
        rook = Piece.WHITE_ROOK if player == 1 else Piece.BLACK_ROOK

        if rights["kingside"]:
            rook_col = bs - 1
            if (self.board[row, rook_col] == rook and
                    all(self.board[row, c] == Piece.EMPTY for c in range(col + 1, rook_col))):
                moves.append(
                    Move((row, col), (row, col + 2), self.board[row, col], is_castling=True)
                )

        if rights["queenside"]:
            rook_col = 0
            if (self.board[row, rook_col] == rook and
                    all(self.board[row, c] == Piece.EMPTY for c in range(rook_col + 1, col))):
                moves.append(
                    Move((row, col), (row, col - 2), self.board[row, col], is_castling=True)
                )

        return moves

    def get_piece_moves(self, row: int, col: int, check_legal: bool = True) -> List[Move]:
        moves = super().get_piece_moves(row, col, check_legal)
        if not check_legal:
            return moves
        result = []
        for move in moves:
            if move.is_castling:
                if self._is_in_check(self.current_player):
                    continue
                through_col = (move.start[1] + move.end[1]) // 2
                if self._is_square_attacked(move.start[0], through_col, 3 - self.current_player):
                    continue
            result.append(move)
        return result
