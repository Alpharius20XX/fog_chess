# fog_chess
Based on https://github.com/Devanik21/MiniChess-RL/tree/main adapted to fog of war chess


# Setup

```
conda create -n fog_chess python=3.12
conda activate fog_chess
pip install -r requirements.txt
```

# Train and play

Train a fog-observation Q-learning model:

```
python -m fog_chess.rl --episodes 5000 --seed 7 --report-every 100 --eval-games 300 --log-file logs/fog_q_learning5000.log --model-file models/fog_q_learning.json
```

The Q-learning state uses each player's memory board: visible squares update the
memory, and hidden squares keep the last remembered piece or empty square.

Play against the trained model:

```
streamlit run fog_chess/app.py
```

The UI loads `models/fog_q_learning.json` by default. After running the Streamlit
command, open the local URL it prints, choose your side, and enter moves like
`a2a3` or `a2a1=Q`. You can also move by clicking a piece and then a destination
square. The board highlights recent moves and fog-visibility changes, and shows
captured pieces for both sides. Use `Undo last turn` to take back your previous
move and the AI response.