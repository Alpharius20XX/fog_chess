from .chess import Minichess
from .visual import visualize_board


def play_game(env, agent1, agent2, training=True, verbose=False):
    env.reset()
    history = []
    agents = {1: agent1, 2: agent2}

    while not env.game_over and env.move_count < 100:
        player = env.current_player
        agent = agents[player]
        state = env.get_state()
        move = agent.choose_action(env, training=training)
        if move is None:
            break

        history.append((state, move, player))
        env.make_move(move)

        if verbose:
            name = "White" if player == 1 else "Black"
            print(f"{env.move_count}. {name}: {move.to_notation()}")
            env.print_board()

    if env.winner == 1:
        agent1.wins += 1
        agent2.losses += 1
    elif env.winner == 2:
        agent2.wins += 1
        agent1.losses += 1
    else:
        agent1.draws += 1
        agent2.draws += 1

    if training:
        agent1.update_from_game(history, env.winner or 0)
        agent2.update_from_game(history, env.winner or 0)

    return env.winner


def parse_move_input(text, valid_moves):
    """
    User input format:
    e.g. a2a3, b1c3, a2a1=Q
    """
    text = text.strip().lower().replace(" ", "")
    cols = "abcde"

    for move in valid_moves:
        if move.to_notation().lower().replace("+", "").replace("#", "") == text:
            return move

    # allow simple input like a2a3 without promotion/check signs
    if len(text) >= 4:
        try:
            sc = cols.index(text[0])
            sr = 5 - int(text[1])
            ec = cols.index(text[2])
            er = 5 - int(text[3])

            promotion = None
            if "=" in text:
                promotion = text.split("=")[1].upper()

            for move in valid_moves:
                if move.start == (sr, sc) and move.end == (er, ec):
                    if promotion is None or (
                        move.promotion is not None
                        and move.promotion.to_string().upper() == promotion
                    ):
                        return move
        except Exception:
            return None

    return None


def play_two_players(visual=False):
    game = Minichess()

    print("Game started.")
    print("Input move like: a2a3 or b1c3")

    last_move = None

    if visual:
        visualize_board(game.board, "Initial Board", last_move)

    while not game.game_over and game.move_count < 100:
        player = game.current_player
        name = "White" if player == 1 else "Black"

        print(f"{name} to play")

        valid_moves = game.get_all_valid_moves()

        print("Your valid moves:")
        print(", ".join(m.to_notation() for m in valid_moves))

        user_input = input("Your move: ")
        move = parse_move_input(user_input, valid_moves)

        if move is None:
            print("Invalid move. Try again.")
            continue

        game.make_move(move)
        last_move = move
        print(f"You played: {move.to_notation()}")

        if visual:
            visualize_board(
                game.board,
                f"Move {game}: {name} {last_move.to_notation()}",
                last_move,
            )

    print(f"Winner: {game.winner}")


if __name__ == "__main__":
    play_two_players()
