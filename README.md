# Fog of War Mini-Chess
Based on https://github.com/Devanik21/MiniChess-RL/tree/main adapted to fog of war chess


# Setup

```
conda create -n fog_chess python=3.12
conda activate fog_chess
pip install -r requirements.txt
```

# Introduction
Fog of war is a variant of chess where each player can only see their own pieces, and squares where their own pieces can move to (including by capturing). This turns chess from a perfect to an imperfect information game. 
Mini-chess is a further variant of chess where the board is smaller: 5x5 instead of 8x8, and fewer pieces are used. We will use this variant because we hope it will make the game easier to learn for a reinforcement learning agent. 
We call the combination of both variants FOWMC. 

```
k q b n r
p p p p p
. . . . .
P P P P P
K Q B N R
```

## The State Space
The board in chess can be represented as a matrix with an enum value at each cell representing a piece (king, queen, rook, knight, bishop, pawn), and empty. The state space can be a square matrix, where in each square an enum index is included. There is an enum index for each type of piece in each team, for empty squares, and for unknown squares.

We can also represent the state in an unstructured way. Instead of a matrix, we have a set of squares. Each square is a tuple containing the location and kind of piece. The location can be represented by the row and column, or by an index. Additionally, this tuple could contain the turn number. This would allow us to represent the current state with a set of all historic revealed pieces, and the set of unknown pieces in the latest turn. For example:

s = (t, p, k)
t is the turn
p is an enum with the square position
k is an enum with the kind of piece / empty / unknown
our full state representation is a set of s: {s0, s1, s2, …}

## The Action Space
We can represent the action space as a set of legal state transitions. Each transition is represented by a tuple containing the location of the piece to move, and the location of the destination. 
By using a set, we can always easily represent all the legal actions the agent can take, and they can be transformed with an attention mechanism into probabilities. 
