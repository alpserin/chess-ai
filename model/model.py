# model/train.py
import chess.pgn
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import string

class MoveEncoder:
    def __init__(self):
        self.square_to_index = {square: i for i, square in enumerate([f"{file}{rank}" for file in string.ascii_lowercase[:8] for rank in range(1, 9)])}
        self.promotion_pieces = {'q': 1, 'r': 2, 'b': 3, 'n': 4}
    
    def encode_move(self, move):
        if len(move) < 4 or len(move) > 5:
            print(f"Invalid move length: {move}")
            return 0
        
        from_square = move[:2]
        to_square = move[2:4]
        
        if from_square not in self.square_to_index or to_square not in self.square_to_index:
            print(f"Invalid squares in move: {move}")
            return 0
        
        from_index = self.square_to_index[from_square]
        to_index = self.square_to_index[to_square]
        
        promotion_piece = self.promotion_pieces.get(move[4], 0) if len(move) == 5 else 0
        
        move_index = from_index * 64 * 5 + to_index * 5 + promotion_piece
        
        return move_index
    
    def decode_move(self, move_index):
        from_index = move_index // (64 * 5)
        to_index = (move_index // 5) % 64
        promotion_piece = move_index % 5
        
        from_square = list(self.square_to_index.keys())[from_index]
        to_square = list(self.square_to_index.keys())[to_index]
        
        promotion_piece = '' if promotion_piece == 0 else list(self.promotion_pieces.keys())[promotion_piece - 1]
        
        return f"{from_square}{to_square}{promotion_piece}"

encoder = MoveEncoder()

def parse_pgn(pgn_file):
    games = []
    with open(pgn_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
    return games

def extract_features(game):
    headers = game.headers
    white_elo = int(headers.get("WhiteElo", 1500))
    black_elo = int(headers.get("BlackElo", 1500))
    result = headers.get("Result", "1/2-1/2")
    result = 1 if result == "1-0" else 0 if result == "0-1" else 0.5
    
    moves = []
    node = game
    while node.variations:
        next_node = node.variation(0)
        move = node.board().uci(next_node.move)
        if not move:
            print(f"Empty move found in game {game.headers['LichessURL']}")
        moves.append(move)
        node = next_node
    
    return white_elo, black_elo, result, moves

class ChessDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.max_len = max(dataframe['Moves'].apply(len))
        self.encoder = MoveEncoder()
        
        # Normalizing Elo ratings
        self.white_elo_mean = dataframe['WhiteElo'].mean()
        self.white_elo_std = dataframe['WhiteElo'].std()
        self.black_elo_mean = dataframe['BlackElo'].mean()
        self.black_elo_std = dataframe['BlackElo'].std()
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        white_elo = (row['WhiteElo'] - self.white_elo_mean) / self.white_elo_std
        black_elo = (row['BlackElo'] - self.black_elo_mean) / self.black_elo_std
        result = row['Result']
        
        moves = row['Moves']
        moves_encoded = [self.encoder.encode_move(move) for move in moves]
        moves_encoded += [0] * (self.max_len - len(moves_encoded))
        
        return torch.tensor([white_elo, black_elo]), torch.tensor(moves_encoded), torch.tensor(result)

class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True, dropout=0.5)
        self.fc3 = nn.Linear(128 + 64, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, moves):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        
        moves, _ = self.lstm(moves.unsqueeze(-1).float())
        moves = moves[:, -1, :]  # Get the output of the last LSTM cell
        
        x = torch.cat((x, moves), dim=1)
        x = torch.relu(self.fc3(x))
        x = self.dropout(torch.relu(self.fc4(x)))
        x = torch.sigmoid(self.fc5(x))
        return x
