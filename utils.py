import chess

def promote_pawn(board, move):
    if board.piece_at(move.from_square).piece_type == chess.PAWN:
        if chess.square_rank(move.to_square) in [0, 7]:
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    return move
