# ai.py
import chess
import time
import torch
import torch.nn as nn
from model.train import MoveEncoder, ChessNN

# Load the trained model
model = ChessNN().cuda()
model.load_state_dict(torch.load("chess_model.pth"))
model.eval()

transposition_table = {}

def evaluate_board_nn(board):
    # Encode the board and features for the neural network
    encoder = MoveEncoder()
    white_elo, black_elo = 1500, 1500  # You can adjust these as needed
    moves = [move.uci() for move in board.move_stack]
    moves_encoded = [encoder.encode_move(move) for move in moves]
    moves_encoded = torch.tensor(moves_encoded).cuda().float()
    
    # Normalize ELO ratings
    white_elo = (white_elo - 1500) / 300
    black_elo = (black_elo - 1500) / 300
    
    features = torch.tensor([white_elo, black_elo]).cuda().float().unsqueeze(0)
    moves_encoded = moves_encoded.unsqueeze(0)
    
    with torch.no_grad():
        eval = model(features, moves_encoded).item()
        
    return eval * 2 - 1  # Convert to the range [-1, 1]

def order_moves(board):
    moves = list(board.legal_moves)
    
    # Most Valuable Victim - Least Valuable Attacker (MVV-LVA)
    def mvv_lva(move):
        capture_value = 0
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            attacker_piece = board.piece_at(move.from_square)
            if captured_piece and attacker_piece:
                capture_value = (captured_piece.piece_type * 10) - attacker_piece.piece_type
        return capture_value

    # Moves that give check
    def gives_check(move):
        board.push(move)
        check = board.is_check()
        board.pop()
        return check

    # Promotion moves
    def is_promotion(move):
        return move.promotion is not None

    # Combining the heuristics for ordering
    moves.sort(key=lambda move: (
        gives_check(move),  # Prioritize moves that give check
        is_promotion(move),  # Then prioritize promotion moves
        mvv_lva(move)  # Finally, use MVV-LVA for captures
    ), reverse=True)

    return moves


def minimax(board, depth, alpha, beta, maximizing_player, start_time, time_limit):
    if depth == 0 or board.is_game_over() or time.time() - start_time > time_limit:
        return evaluate_board_nn(board)

    key = chess.polyglot.zobrist_hash(board)
    if key in transposition_table:
        return transposition_table[key]

    legal_moves = order_moves(board)
    if maximizing_player:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, start_time, time_limit)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        transposition_table[key] = max_eval
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, start_time, time_limit)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        transposition_table[key] = min_eval
        return min_eval


def find_best_move(board, max_depth, time_limit):
    best_move = None
    start_time = time.time()
    for depth in range(1, max_depth + 1):
        best_value = float('-inf')
        legal_moves = order_moves(board)
        for move in legal_moves:
            board.push(move)
            board_value = minimax(board, depth - 1, float('-inf'), float('inf'), False, start_time, time_limit)
            board.pop()
            if board_value > best_value:
                best_value = board_value
                best_move = move
            if time.time() - start_time > time_limit:
                break
        if time.time() - start_time > time_limit:
            break
    return best_move
