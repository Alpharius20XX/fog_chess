import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .chess import Piece

_PIECE_SYMBOLS = {
    Piece.WHITE_KING: "♚",
    Piece.WHITE_QUEEN: "♛",
    Piece.WHITE_ROOK: "♜",
    Piece.WHITE_BISHOP: "♝",
    Piece.WHITE_KNIGHT: "♞",
    Piece.WHITE_PAWN: "♟",
    Piece.BLACK_KING: "♚",
    Piece.BLACK_QUEEN: "♛",
    Piece.BLACK_ROOK: "♜",
    Piece.BLACK_BISHOP: "♝",
    Piece.BLACK_KNIGHT: "♞",
    Piece.BLACK_PAWN: "♟",
}


def visualize_board(board, last_move=None, pause=0.8):
    bs = len(board)
    fig, ax = plt.subplots(figsize=(bs, bs))
    for row in range(bs):
        for col in range(bs):
            piece = board[row][col] if isinstance(board[0], tuple) else board[row, col]
            if last_move and (
                (row, col) == last_move.start or (row, col) == last_move.end
            ):
                colour = "#BACA44"
            else:
                colour = "#F0D9B5" if (row + col) % 2 == 0 else "#B58863"
            ax.add_patch(Rectangle((col, bs - 1 - row), 1, 1, facecolor=colour))
            if piece not in (Piece.EMPTY, Piece.UNKNOWN):
                ax.text(
                    col + 0.5,
                    bs - 1 - row + 0.5,
                    _PIECE_SYMBOLS.get(piece, piece.to_string()),  # type: ignore
                    ha="center",
                    va="center",
                    fontsize=34,
                    color="white" if piece.is_white() else "black",
                )
            elif piece == Piece.UNKNOWN:
                ax.text(
                    col + 0.5,
                    bs - 1 - row + 0.5,
                    _PIECE_SYMBOLS.get(piece, piece.to_string()),  # type: ignore
                    ha="center",
                    va="center",
                    fontsize=34,
                    color="#A3A3A3",
                )

    cols_label = "abcdefgh"[:bs]
    for i in range(bs):
        ax.text(-0.25, bs - 1 - i + 0.5, str(bs - i), ha="center", va="center")
        ax.text(i + 0.5, -0.25, cols_label[i], ha="center", va="center")

    ax.set_xlim(-0.5, bs)
    ax.set_ylim(-0.5, bs)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.show(block=False)
    plt.pause(pause)
    plt.close(fig)
