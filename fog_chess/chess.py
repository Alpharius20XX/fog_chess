import argparse
import json
import math
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Move
# ============================================================

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


# ============================================================
# Gardner 5x5 Minichess Environment
# ============================================================

class Minichess:
    PIECE_VALUES = {
        "P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 20000,
        "p": -100, "n": -320, "b": -330, "r": -500, "q": -900, "k": -20000,
    }

    def __init__(self):
        self.board_size = 5
        self.masked=True

        self.show_turn=5

        self.reset()

    def reset(self):
        self.board = np.array([
            ["k", "q", "b", "n", "r"],
            ["p", "p", "p", "p", "p"],
            [".", ".", ".", ".", "."],
            ["P", "P", "P", "P", "P"],
            ["K", "Q", "B", "N", "R"],
        ], dtype=str)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history: List[Move] = []
        self.move_count = 0

        if(self.masked):
            self.w_hist=[]
            self.b_hist=[]
        

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

    def get_piece_moves(self, row: int, col: int, check_legal: bool = True) -> List[Move]:
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
                    moves.append(Move((row, col), (new_row, col), piece, promotion=promo))
            else:
                moves.append(Move((row, col), (new_row, col), piece))

        for dc in [-1, 1]:
            new_col = col + dc
            if 0 <= new_row < 5 and 0 <= new_col < 5:
                target = self.board[new_row, new_col]
                if target != "." and self.is_enemy(target, self.current_player):
                    if new_row == promotion_row:
                        for promo in ["Q", "R", "B", "N"]:
                            moves.append(Move((row, col), (new_row, new_col), piece, captured=target, promotion=promo))
                    else:
                        moves.append(Move((row, col), (new_row, new_col), piece, captured=target))
        return moves

    def _get_knight_moves(self, row: int, col: int) -> List[Move]:
        moves = []
        piece = self.board[row, col]
        for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                       (1, -2), (1, 2), (2, -1), (2, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 5 and 0 <= nc < 5:
                target = self.board[nr, nc]
                if target == "." or self.is_enemy(target, self.current_player):
                    moves.append(Move((row, col), (nr, nc), piece, captured=None if target == "." else target))
        return moves

    def _get_sliding_moves(self, row: int, col: int, directions: List[Tuple[int, int]]) -> List[Move]:
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
        return self._get_sliding_moves(row, col, [
            (-1, -1), (-1, 1), (1, -1), (1, 1),
            (-1, 0), (1, 0), (0, -1), (0, 1),
        ])

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
                        moves.append(Move((row, col), (nr, nc), piece, captured=None if target == "." else target))
        return moves

    def get_all_valid_moves(self,maskedhist=None) -> List[Move]:

        
        moves = []
        if(maskedhist is None):
            for row in range(5):
                for col in range(5):
                    piece = self.board[row, col]
                    if piece != "." and self.is_friendly(piece, self.current_player):

                        moves.extend(self.get_piece_moves(row, col))
        else:#for if we want to check consistency with history
            for row in range(5):
                for col in range(5):
                    piece = self.board[row, col]
                    if piece != "." and self.is_friendly(piece, self.current_player):

                        cand_moves=[]

                        

                        for move in (self.get_piece_moves(row, col)):

                            self._make_move_internal(move= move)
                            if np.array_equal(self.mask(), maskedhist):
                                cand_moves+=[move]

                        moves.extend(cand_moves)
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
        piece = move.promotion if move.promotion and self.current_player == 1 else \
                move.promotion.lower() if move.promotion else self.board[sr, sc]
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

        if(self.masked):
            if(len(self.b_hist)==self.show_turn+1): #show every five turns
                self.w_hist=[self.board]
                self.b_hist=[self.board]
            else:
                self.mask()
            

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
        center_table = np.array([
            [0, 0, 0, 0, 0],
            [0, 5, 5, 5, 0],
            [0, 5, 10, 5, 0],
            [0, 5, 5, 5, 0],
            [0, 0, 0, 0, 0],
        ])

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


    def mask(self):
            
        masked = np.full_like(self.board, '_', dtype=str)

        for r in range(5):
            for c in range(5):
                piece = self.board[r, c]
                if self.is_friendly(piece, self.current_player):

                    for move in self.get_piece_moves(r, c, check_legal=False):
                            masked[move.end]=self.board[move.end]
                    
        if(self.current_player==1):
            self.w_hist+=[masked]

        else:
            self.b_hist+=[masked]

        return masked
    

    







# ============================================================
# MCTS
# ============================================================

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None, prior=1.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: Dict[Move, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        if(self.parent==None):
            depth=0
        else:
            depth=self.parent.depth+1

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def ucb_score(self, parent_visits, c_puct=1.5):
        q = self.value() if self.visit_count > 0 else 0.0
        u = c_puct * self.prior * math.sqrt(parent_visits + 1) / (1 + self.visit_count)
        return q + u

    def select_child(self, c_puct=1.5):
        return max(self.children.values(), key=lambda child: child.ucb_score(self.visit_count, c_puct))

    def expand(self, game, policy_priors,maskedhist=None,move_hist=None):
        valid_moves = game.get_all_valid_moves(maskedhist=maskedhist)
        if not valid_moves:
            return
        total_prior = sum(policy_priors.values()) or len(valid_moves)
        for move in valid_moves:
            child_game = game.copy()
            child_game.make_move(move)
            prior = policy_priors.get(move, 1.0) / total_prior
            self.children[move] = MCTSNode(child_game.get_state(), parent=self, move=move, prior=prior)
            if not (maskedhist is None):
                child_game = game.copy()
                child_game.make_move(move_hist[-(len(maskedhist)*2-1)])
                prior = policy_priors.get(move, 1.0) / total_prior
                self.children[move] = MCTSNode(child_game.get_state(), parent=self, move=move, prior=prior)
        self.is_expanded = True

    def backup(self, value):
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)


# ============================================================
# Agent
# ============================================================

class Agent:
    def __init__(self, player_id, lr=0.3, gamma=0.99, epsilon=1.0):
        self.player_id = player_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.96
        self.epsilon_min = 0.01
        self.mcts_simulations = 100
        self.c_puct = 1.4
        self.minimax_depth = 3
        self.policy_table = defaultdict(lambda: defaultdict(float))
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.hidden_depth=0
        self.youngest_node=None
        self.oldest_node=None

    def get_policy_priors(self, game):
        state = game.get_state()
        priors = {}
        for move in game.get_all_valid_moves():
            if move in self.policy_table[state]:
                priors[move] = self.policy_table[state][move]
            else:
                prior = 1.0
                if move.captured:
                    prior += abs(Minichess.PIECE_VALUES.get(move.captured, 0)) / 100
                if move.promotion:
                    prior += 3.0
                priors[move] = prior
        return priors
    
    def mcts_histsearch(self, game,maskedhist):

        move_hist=game.move_history


        if(len(maskedhist)<self.hidden_depth):
            self.hidden_depth=0
            self.oldest_node=MCTSNode(game.get_state())


        root = self.oldest_node


        maxdepth=self.hidden_depth

        while maxdepth<len(maskedhist):
            node = root
            search_game = game.copy()

            while node.is_expanded and node.children:
                node = node.select_child(self.c_puct)

                if(node.depth>maxdepth):
                    maxdepth=node.depth

                search_game.make_move(node.move)

                

            if not search_game.game_over:
                node.expand(search_game, self.get_policy_priors(search_game),maskedhist[node.depth+1:],move_hist=move_hist)

            

            value = self._evaluate_leaf(search_game)
            node.backup(value)
            self.youngest_node=node #could make more efficient

        

        self.hidden_depth=maxdepth

        return root

    def mcts_search(self, game, num_simulations):

        root = MCTSNode(game.get_state())

        for _ in range(num_simulations):
            node = root
            search_game = game.copy()

            while node.is_expanded and node.children:
                node = node.select_child(self.c_puct)
                search_game.make_move(node.move)

            if not search_game.game_over:
                node.expand(search_game, self.get_policy_priors(search_game))

            value = self._evaluate_leaf(search_game)
            node.backup(value)

        return root

    def _evaluate_leaf(self, game):
        if game.game_over:
            if game.winner == self.player_id:
                return 1.0
            if game.winner == 3 - self.player_id:
                return -1.0
            return 0.0
        score = self._minimax(game, self.minimax_depth, -float("inf"), float("inf"), True)
        return float(np.tanh(score / 1000))

    def _minimax(self, game, depth, alpha, beta, maximizing):
        if depth == 0 or game.game_over:
            return game.evaluate_position(self.player_id)

        moves = game.get_all_valid_moves()
        if not moves:
            return game.evaluate_position(self.player_id)
        
        #removing the random move truncation

        #moves = moves[:8]
        if maximizing:
            best = -float("inf")
            for move in moves:
                sim = game.copy()
                sim.make_move(move)
                val = self._minimax(sim, depth - 1, alpha, beta, False)
                best = max(best, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return best
        else:
            best = float("inf")
            for move in moves:
                sim = game.copy()
                sim.make_move(move)
                val = self._minimax(sim, depth - 1, alpha, beta, True)
                best = min(best, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return best

    def choose_action(self, game, training=True):#modify to do fog


        moves = game.get_all_valid_moves()
        if not moves:
            return None

        if training and random.random() < self.epsilon:
            return random.choice(moves)
        
        #fog stuff

        fake_game=self.hidden_sample(game)

        root = self.mcts_search(fake_game, self.mcts_simulations)
        if not root.children:
            return random.choice(moves)

        best_move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]

        state = game.get_state()
        total_visits = sum(child.visit_count for child in root.children.values()) or 1
        for move, child in root.children.items():
            self.policy_table[state][move] = child.visit_count / total_visits

        return best_move
    
    def hidden_sample(self,real_game):

        game=real_game.copy()


        if(game.current_player==1):
            hist=game.w_hist
        else:
            hist=game.b_hist
        
        game.board=hist[0]

        if(len(hist)==1):
            return(game)

        game.current_player=3-game.current_player

        root = self.mcts_histsearch(game,hist)

        Node=self.youngest_node

        moves=[]

        while(Node.parent!=None):
            moves+=[Node.move]
            Node=Node.parent

        
        moves=moves[::-1]

        for move in moves:
            game.make_move(move)

        


        #best_move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]##dubious

        

        game.current_player=3-game.current_player



        return game








    def update_from_game(self, game_data, result):
        for state, move, player in game_data:
            if player != self.player_id:
                continue
            reward = 1.0 if result == self.player_id else 0.0 if result == 0 else -1.0
            current = self.policy_table[state][move]
            self.policy_table[state][move] = current + self.lr * (reward - current)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_stats(self):
        self.wins = self.losses = self.draws = 0


        


# ============================================================
# Training and Play
# ============================================================

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


def train(episodes=100, mcts1=50, mcts2=50, depth1=2, depth2=2):
    env = Minichess()
    agent1 = Agent(1)
    agent2 = Agent(2)
    agent1.mcts_simulations = mcts1
    agent2.mcts_simulations = mcts2
    agent1.minimax_depth = depth1
    agent2.minimax_depth = depth2

    for ep in range(1, episodes + 1):
        winner = play_game(env, agent1, agent2, training=True)
        agent1.decay_epsilon()
        agent2.decay_epsilon()

        if ep % max(1, episodes // 10) == 0:
            print(
                f"Episode {ep}/{episodes} | "
                f"White wins={agent1.wins}, Black wins={agent2.wins}, Draws={agent1.draws}, "
                f"eps=({agent1.epsilon:.3f}, {agent2.epsilon:.3f}), "
                f"policies=({len(agent1.policy_table)}, {len(agent2.policy_table)})"
            )

    return agent1, agent2


# ============================================================
# Matplotlib Visualisation
# ============================================================

def visualize_board(board, title="Minichess Board", last_move=None, pause=0.8):
    piece_symbols = {
        "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
        "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
    }

    fig, ax = plt.subplots(figsize=(5, 5))
    for row in range(5):
        for col in range(5):
            colour = "#F0D9B5" if (row + col) % 2 == 0 else "#B58863"
            if last_move and ((row, col) == last_move.start or (row, col) == last_move.end):
                colour = "#BACA44"
            ax.add_patch(plt.Rectangle((col, 4 - row), 1, 1, facecolor=colour))
            piece = board[row, col]
            if piece != ".":
                ax.text(col + 0.5, 4 - row + 0.5, piece_symbols.get(piece, piece),
                        ha="center", va="center", fontsize=34,
                        color="white" if piece.isupper() else "black")

    for i in range(5):
        ax.text(-0.25, 4 - i + 0.5, str(5 - i), ha="center", va="center")
        ax.text(i + 0.5, -0.25, "abcde"[i], ha="center", va="center")

    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    plt.show(block=False)
    plt.pause(pause)
    plt.close(fig)


def watch_game(agent1, agent2, visual=True):
    env = Minichess()
    agents = {1: agent1, 2: agent2}

    print("Initial board:")
    # env.print_board()
    if visual:
        visualize_board(env.board, "Initial Board", pause=1.0)

    while not env.game_over and env.move_count < 100:
        player = env.current_player
        move = agents[player].choose_action(env, training=False)
        if move is None:
            break

        env.make_move(move)
        name = "White" if player == 1 else "Black"
        print(f"{env.move_count}. {name}: {move.to_notation()}")
        # env.print_board()

        if visual:
            visualize_board(env.board, f"Move {env.move_count}: {name} {move.to_notation()}", move)

    if env.winner == 1:
        print("White wins")
    elif env.winner == 2:
        print("Black wins")
    else:
        print("Draw")

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
                    if promotion is None or move.promotion == promotion:
                        return move
        except Exception:
            return None

    return None


def play_human_vs_agent(agent, human_player=1, visual=True):
    env = Minichess()
    agent.player_id = 3 - human_player

    print("Game started.")
    print("You are", "White" if human_player == 1 else "Black")
    print("Input move like: a2a3 or b1c3")

    last_move = None

    if visual:
        visualize_board(env.board, "Initial Board", last_move)

    while not env.game_over and env.move_count < 100:
        player = env.current_player
        name = "White" if player == 1 else "Black"

        if env.current_player == human_player:
            valid_moves = env.get_all_valid_moves()

            print("Your valid moves:")
            print(", ".join(m.to_notation() for m in valid_moves))

            user_input = input("Your move: ")
            move = parse_move_input(user_input, valid_moves)

            if move is None:
                print("Invalid move. Try again.")
                continue

            env.make_move(move)
            last_move = move
            print(f"You played: {move.to_notation()}")

        else:
            print("Agent thinking...")
            move = agent.choose_action(env, training=False)

            if move is None:
                break

            env.make_move(move)
            last_move = move
            print(f"Agent plays: {move.to_notation()}")

        if visual:
            visualize_board(
                env.board,
                f"Move {env.move_count}: {name} {last_move.to_notation()}",
                last_move
            )

    if env.winner == human_player:
        print("You win!")
    elif env.winner == 3 - human_player:
        print("Agent wins!")
    else:
        print("Draw.")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Minichess RL without Streamlit")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--mcts1", type=int, default=50)
    parser.add_argument("--mcts2", type=int, default=50)
    parser.add_argument("--depth1", type=int, default=2)
    parser.add_argument("--depth2", type=int, default=2)
    parser.add_argument("--no-visual", action="store_true")
    args = parser.parse_args()

    agent1, agent2 = train(
        episodes=args.episodes,
        mcts1=args.mcts1,
        mcts2=args.mcts2,
        depth1=args.depth1,
        depth2=args.depth2,
    )

    print("\nTraining complete. Watching one game...\n")
    watch_game(agent1, agent2, visual=not args.no_visual)


if __name__ == "__main__":
    main()
