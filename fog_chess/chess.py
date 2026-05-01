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

    def to_notation(self) -> str:
        cols = "abcde"
        s = f"{cols[self.start[1]]}{5 - self.start[0]}"
        e = f"{cols[self.end[1]]}{5 - self.end[0]}"
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
    def __init__(self):
        self.board_size = 5
        self.reset()

    def reset(self, fog: bool = False):
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
        return self.get_observation() if fog else self.get_state()

    def get_state(self):
        return tuple(tuple(row) for row in self.board)

    def get_observation(self, player: Optional[int] = None):
        return self.get_fog_state(self.current_player if player is None else player)

    def get_fog_state(self, player: int):
        visible: set = set()
        original_player = self.current_player
        self.current_player = player
        try:
            for row in range(5):
                for col in range(5):
                    piece = self.board[row, col]
                    if piece != Piece.EMPTY and self.is_friendly(piece, player):
                        visible.add((row, col))
                        for move in self.get_piece_moves(row, col, check_legal=False):
                            visible.add(move.end)
        finally:
            self.current_player = original_player

        fog_board = np.empty((5, 5), dtype=object)
        for row in range(5):
            for col in range(5):
                fog_board[row, col] = self.board[row, col] if (row, col) in visible else Piece.UNKNOWN
        return tuple(tuple(r) for r in fog_board)

    def copy(self):
        new_game = Minichess()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.move_history = self.move_history.copy()
        new_game.move_count = self.move_count
        return new_game

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

        if self.current_player == 1:
            direction = -1
            promotion_row = 0
            promotions = _WHITE_PROMOTIONS
        else:
            direction = 1
            promotion_row = 4
            promotions = _BLACK_PROMOTIONS

        new_row = row + direction

        if 0 <= new_row < 5 and self.board[new_row, col] == Piece.EMPTY:
            if new_row == promotion_row:
                for promo in promotions:
                    moves.append(
                        Move((row, col), (new_row, col), piece, promotion=promo)
                    )
            else:
                moves.append(Move((row, col), (new_row, col), piece))

        for dc in [-1, 1]:
            new_col = col + dc
            if 0 <= new_row < 5 and 0 <= new_col < 5:
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
            if 0 <= nr < 5 and 0 <= nc < 5:
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
        for dr, dc in directions:
            for i in range(1, 5):
                nr, nc = row + dr * i, col + dc * i
                if not (0 <= nr < 5 and 0 <= nc < 5):
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
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < 5 and 0 <= nc < 5:
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
        for row in range(5):
            for col in range(5):
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
            for r in range(5):
                for c in range(5):
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

    def make_move(self, move: Move):
        if self.game_over:
            return self.get_state(), 0, True

        reward = 0
        if move.captured:
            if move.captured in (Piece.WHITE_KING, Piece.BLACK_KING):
                reward = 500
            else:
                reward = abs(move.captured.get_value(True)) / 100
        if move.promotion:
            reward += 5

        self._make_move_internal(move)
        self.move_history.append(move)
        self.move_count += 1
        self.current_player = 3 - self.current_player

        opponent_moves = self.get_all_valid_moves()
        is_check = self._is_in_check(self.current_player)

        if not opponent_moves:
            self.game_over = True
            if is_check:
                self.winner = 3 - self.current_player
                reward = 500
            else:
                self.winner = 0
                reward = 0

        if self.move_count >= 100:
            self.game_over = True
            self.winner = 0

        return self.get_state(), reward, self.game_over

    def step(self, move: Move, fog: bool = True):
        player = self.current_player
        _, reward, done = self.make_move(move)
        observation = self.get_observation() if fog else self.get_state()
        info = {
            "player": player,
            "next_player": self.current_player,
            "winner": self.winner,
            "move_count": self.move_count,
        }
        return observation, reward, done, info

    def evaluate_position(self, player: int) -> float:
        if self.winner == player:
            return 100000
        if self.winner == 3 - player:
            return -100000
        if self.winner == 0:
            return 0

        score = 0
        center_table = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 5, 5, 5, 0],
                [0, 5, 10, 5, 0],
                [0, 5, 5, 5, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        for row in range(5):
            for col in range(5):
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
        print("\n  a b c d e")
        for r in range(5):
            row_str = " ".join(p.to_string() for p in self.board[r])
            print(f"{5-r} {row_str} {5-r}")
        print("  a b c d e\n")
