import pygame
import chess

# Colors
WHITE = pygame.Color('white')
BLACK = pygame.Color('gray')
BLUE = pygame.Color('blue')

# Load images
IMAGES = {}

def load_images(SQ_SIZE):
    pieces = ['wp', 'wn', 'wb', 'wr', 'wq', 'wk', 'bp', 'bn', 'bb', 'br', 'bq', 'bk']
    for piece in pieces:
        IMAGES[piece] = pygame.transform.scale(pygame.image.load(f"images/{piece}.png"), (SQ_SIZE, SQ_SIZE))

def draw_board(screen, SQ_SIZE):
    colors = [pygame.Color('white'), pygame.Color('gray')]
    for r in range(8):
        for c in range(8):
            color = colors[((r+c) % 2)]
            pygame.draw.rect(screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(screen, board, SQ_SIZE):
    for r in range(8):
        for c in range(8):
            piece = board.piece_at(chess.square(c, 8 - r - 1))
            if piece:
                piece_image = f"{'w' if piece.color else 'b'}{piece.symbol().lower()}"
                screen.blit(IMAGES[piece_image], pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
