from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class Move:
    start: Tuple[int, int]
    end: Tuple[int, int]
    piece: str
    captured: Optional[str] = None
    promotion: Optional[str] = None
    is_check: bool = False
    is_checkmate: bool = False

    def to_notation(self) -> str:
        cols = "abcde"
        s = f"{cols[self.start[1]]}{5 - self.start[0]}"
        e = f"{cols[self.end[1]]}{5 - self.end[0]}"
        notation = f"{s}{e}"
        if self.promotion:
            notation += f"={self.promotion.upper()}"
        if self.is_checkmate:
            notation += "#"
        elif self.is_check:
            notation += "+"
        return notation


class Minichess:
    PIECE_VALUES = {
        "P": 100,
        "N": 320,
        "B": 330,
        "R": 500,
        "Q": 900,
        "K": 20000,
        "p": -100,
        "n": -320,
        "b": -330,
        "r": -500,
        "q": -900,
        "k": -20000,
    }

    def __init__(self):
        self.board_size = 5
        self.reset()

    def reset(self):
        self.board = np.array(
            [
                ["k", "q", "b", "n", "r"],
                ["p", "p", "p", "p", "p"],
                [".", ".", ".", ".", "."],
                ["P", "P", "P", "P", "P"],
                ["K", "Q", "B", "N", "R"],
            ],
            dtype=str,
        )
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history: List[Move] = []
        self.move_count = 0
        return self.get_state()

    def get_state(self):
        return tuple(tuple(str(c) for c in row) for row in self.board)

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
    def is_white_piece(piece: str) -> bool:
        return piece != "." and piece.isupper()

    @staticmethod
    def is_black_piece(piece: str) -> bool:
        return piece != "." and piece.islower()

    def is_enemy(self, piece: str, player: int) -> bool:
        return self.is_black_piece(piece) if player == 1 else self.is_white_piece(piece)

    def is_friendly(self, piece: str, player: int) -> bool:
        return self.is_white_piece(piece) if player == 1 else self.is_black_piece(piece)

    def get_piece_moves(
        self, row: int, col: int, check_legal: bool = True
    ) -> List[Move]:
        piece = self.board[row, col]
        if piece == "." or not self.is_friendly(piece, self.current_player):
            return []

        piece_type = piece.upper()
        if piece_type == "P":
            moves = self._get_pawn_moves(row, col)
        elif piece_type == "N":
            moves = self._get_knight_moves(row, col)
        elif piece_type == "B":
            moves = self._get_bishop_moves(row, col)
        elif piece_type == "R":
            moves = self._get_rook_moves(row, col)
        elif piece_type == "Q":
            moves = self._get_queen_moves(row, col)
        elif piece_type == "K":
            moves = self._get_king_moves(row, col)
        else:
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
        else:
            direction = 1
            promotion_row = 4

        new_row = row + direction

        if 0 <= new_row < 5 and self.board[new_row, col] == ".":
            if new_row == promotion_row:
                for promo in ["Q", "R", "B", "N"]:
                    moves.append(
                        Move((row, col), (new_row, col), piece, promotion=promo)
                    )
            else:
                moves.append(Move((row, col), (new_row, col), piece))

        for dc in [-1, 1]:
            new_col = col + dc
            if 0 <= new_row < 5 and 0 <= new_col < 5:
                target = self.board[new_row, new_col]
                if target != "." and self.is_enemy(target, self.current_player):
                    if new_row == promotion_row:
                        for promo in ["Q", "R", "B", "N"]:
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
                if target == "." or self.is_enemy(target, self.current_player):
                    moves.append(
                        Move(
                            (row, col),
                            (nr, nc),
                            piece,
                            captured=None if target == "." else target,
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
                if target == ".":
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
                    if target == "." or self.is_enemy(target, self.current_player):
                        moves.append(
                            Move(
                                (row, col),
                                (nr, nc),
                                piece,
                                captured=None if target == "." else target,
                            )
                        )
        return moves

    def get_all_valid_moves(self) -> List[Move]:
        moves = []
        for row in range(5):
            for col in range(5):
                piece = self.board[row, col]
                if piece != "." and self.is_friendly(piece, self.current_player):
                    moves.extend(self.get_piece_moves(row, col))
        return moves

    def _find_king(self, player: int):
        king = "K" if player == 1 else "k"
        positions = np.argwhere(self.board == king)
        return tuple(positions[0]) if len(positions) else None

    def _is_square_attacked(self, row: int, col: int, by_player: int) -> bool:
        original_player = self.current_player
        self.current_player = by_player
        try:
            for r in range(5):
                for c in range(5):
                    piece = self.board[r, c]
                    if piece != "." and self.is_friendly(piece, by_player):
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
        piece = (
            move.promotion
            if move.promotion and self.current_player == 1
            else move.promotion.lower() if move.promotion else self.board[sr, sc]
        )
        self.board[er, ec] = piece
        self.board[sr, sc] = "."

    def make_move(self, move: Move):
        if self.game_over:
            return self.get_state(), 0, True

        reward = 0
        if move.captured:
            reward = abs(self.PIECE_VALUES.get(move.captured, 0)) / 100
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
                reward = 100
            else:
                self.winner = 0
                reward = 0

        if self.move_count >= 100:
            self.game_over = True
            self.winner = 0

        return self.get_state(), reward, self.game_over

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
                if piece == ".":
                    continue
                multiplier = 1 if self.is_friendly(piece, player) else -1
                score += multiplier * abs(self.PIECE_VALUES.get(piece, 0))
                if piece.upper() in ["B", "R", "Q", "K"]:
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
            print(f"{5-r} " + " ".join(self.board[r]) + f" {5-r}")
        print("  a b c d e\n")
