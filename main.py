import pygame
import chess
import time
from draw import draw_board, draw_pieces, load_images
from ai import find_best_move  # Ensure this import matches the function definition in ai.py
from utils import promote_pawn

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
DIMENSION = 8  # Chessboard dimensions are 8x8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15

# Initialize font
pygame.font.init()
font = pygame.font.SysFont("Arial", 32)

def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Chess')
    clock = pygame.time.Clock()
    screen.fill(pygame.Color('white'))
    board = chess.Board()
    load_images(SQ_SIZE)
    running = True
    sq_selected = ()
    player_clicks = []
    depth = 1
    time_limit = 2
    game_over = False
    winner = None

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN and not game_over:
                location = pygame.mouse.get_pos()
                col = location[0] // SQ_SIZE
                row = location[1] // SQ_SIZE
                if sq_selected == (row, col):
                    sq_selected = ()
                    player_clicks = []
                else:
                    sq_selected = (row, col)
                    player_clicks.append(sq_selected)
                if len(player_clicks) == 2:
                    move = chess.Move(chess.square(player_clicks[0][1], DIMENSION - player_clicks[0][0] - 1),
                                      chess.square(player_clicks[1][1], DIMENSION - player_clicks[1][0] - 1))
                    if move in board.legal_moves:
                        board.push(move)
                    else:
                        try:
                            move_uci = move.uci() + "q"  # Append 'q' for promotion
                            if chess.Move.from_uci(move_uci) in board.legal_moves:
                                move = chess.Move.from_uci(move_uci)
                                board.push(move)
                        except ValueError:
                            pass
                    
                    sq_selected = ()
                    player_clicks = []

        if not game_over:
            if board.turn == chess.BLACK and not board.is_game_over():
                ai_move = find_best_move(board, depth, time_limit)
                if ai_move is not None:
                    if board.is_legal(ai_move):
                        board.push(ai_move)
                    else:
                        try:
                            ai_move = promote_pawn(board, ai_move)  # Promote pawn for AI moves
                            board.push(ai_move)
                        except ValueError:
                            pass

            if board.is_checkmate():
                game_over = True
                winner = "Black wins!" if board.turn == chess.WHITE else "White wins!"

        draw_board(screen, SQ_SIZE)
        draw_pieces(screen, board, SQ_SIZE)
        clock.tick(MAX_FPS)
        pygame.display.flip()

        if game_over:
            text_surface = font.render(winner, True, pygame.Color('red'))
            text_rect = text_surface.get_rect(center=(WIDTH//2, HEIGHT//2))
            screen.blit(text_surface, text_rect)
            pygame.display.flip()
            time.sleep(5)
            running = False

if __name__ == "__main__":
    main()
