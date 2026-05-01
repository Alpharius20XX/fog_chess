from .chess import Minichess
from .visual import visualize_board


def parse_move_input(text, valid_moves, board_size):
    """
    User input format: two space-separated square indices, e.g. "20 15"
    Index encoding: row * board_size + col (same as get_legal_move_indices).
    """
    try:
        parts = text.strip().split()
        if len(parts) == 2:
            from_sq, to_sq = int(parts[0]), int(parts[1])
            for move in valid_moves:
                if move.start[0] * board_size + move.start[1] == from_sq and move.end[0] * board_size + move.end[1] == to_sq:
                    return move
    except Exception:
        pass
    return None


def play_two_players(visual=False):
    game = Minichess()
    bs = game.board_size

    print("Game started.")
    print(f"Input move as two indices, e.g. '20 15' (from_sq to_sq, index = row*{bs}+col)")

    last_moves = {}

    while not game.game_over and game.move_count < game.max_moves:
        player = game.current_player
        name = "White" if player == 1 else "Black"

        print(f"{name} to play")

        state = game.get_fog_state(player)
        if visual:
            visualize_board(state, last_moves.get(player))

        valid_moves = game.get_all_valid_moves()
        legal_indices = game.get_legal_move_indices(player)

        print("Your valid moves:")
        print(", ".join(f"({f} {t})" for f, t in legal_indices))

        user_input = input("Your move: ")
        move = parse_move_input(user_input, valid_moves, bs)

        if move is None:
            print("Invalid move. Try again.")
            continue

        game.make_move(move)
        last_moves[player] = move
        print(f"You played: {move.to_notation(bs)}")

    print(f"Winner: {game.winner}")


if __name__ == "__main__":
    play_two_players()
