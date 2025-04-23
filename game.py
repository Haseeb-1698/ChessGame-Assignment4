import pygame
import os
import platform
import asyncio
from abc import ABC, abstractmethod
import math
import chess

# Constants
SQUARE_SIZE = 100
BOARD_SIZE = 8 * SQUARE_SIZE
HISTORY_WIDTH = 200
WIDTH = BOARD_SIZE + HISTORY_WIDTH
HEIGHT = BOARD_SIZE
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (200, 200, 200)
DARK_SQUARE = (0, 0, 0)
HIGHLIGHT = (255, 255, 0, 100)
WIN_OVERLAY = (0, 0, 0, 200)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Game: Human (White) vs AI (Black)")
font = pygame.font.Font(None, 24)
win_font = pygame.font.Font(None, 48)

class Piece:
    def __init__(self, color, symbol):
        self.color = color
        self.symbol = symbol
        try:
            self.image = pygame.image.load(os.path.join("Assets", f"{self.color}{self.symbol}.png"))
            self.image = pygame.transform.scale(self.image, (SQUARE_SIZE, SQUARE_SIZE))
        except pygame.error as e:
            print(f"Error loading image for {self.color}{self.symbol}: {e}")
            self.image = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
            self.image.fill(BLACK if self.color == "b" else WHITE)

class Board:
    def __init__(self):
        self.chess_board = chess.Board()
        self.piece_map = {}
        self.setup_piece_map()

    def setup_piece_map(self):
        for square in chess.SQUARES:
            piece = self.chess_board.piece_at(square)
            if piece:
                color = 'w' if piece.color == chess.WHITE else 'b'
                symbol = piece.symbol().upper()
                if (color, symbol) not in self.piece_map:
                    self.piece_map[(color, symbol)] = Piece(color, symbol)

    def make_move(self, move):
        self.chess_board.push(move)
        if move.promotion:
            square = move.to_square
            piece = self.chess_board.piece_at(square)
            color = 'w' if piece.color == chess.WHITE else 'b'
            symbol = piece.symbol().upper()
            self.piece_map[(color, symbol)] = Piece(color, symbol)

    def undo_move(self):
        self.chess_board.pop()

    def is_in_check(self, color):
        return self.chess_board.is_check()

    def get_all_moves(self, color):
        return list(self.chess_board.legal_moves)

class Evaluation:
    def evaluate(self, board, color):
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        score = 0
        material = {"w": 0, "b": 0}
        for square in chess.SQUARES:
            piece = board.chess_board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                piece_color = 'w' if piece.color == chess.WHITE else 'b'
                material[piece_color] += value
                if piece_color == color:
                    score += value
                else:
                    score -= value
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        for square in center_squares:
            piece = board.chess_board.piece_at(square)
            if piece and piece.color == (chess.WHITE if color == 'w' else chess.BLACK):
                score += 0.5
        return score, material

class Player(ABC):
    def __init__(self, color):
        self.color = color

    @abstractmethod
    async def get_move(self, game):
        pass

class HumanPlayer(Player):
    async def get_move(self, game):
        if not game.running:
            print("Game is over, cannot make moves")
            return None
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Quit event detected in HumanPlayer")
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    row, col = pos[1] // SQUARE_SIZE, pos[0] // SQUARE_SIZE
                    if 0 <= row < 8 and 0 <= col < 8:
                        square = chess.square(col, 7 - row)
                        print(f"Clicked square: ({row}, {col}) -> chess square {square} ({chess.square_name(square)})")
                        if not game.selected:
                            piece = game.board.chess_board.piece_at(square)
                            if piece and piece.color == (chess.WHITE if self.color == 'w' else chess.BLACK):
                                game.selected = (row, col)
                                game.valid_moves = []
                                legal_moves = game.board.get_all_moves(self.color)
                                print(f"Legal moves for {self.color}: {[move.uci() for move in legal_moves]}")
                                for move in legal_moves:
                                    if move.from_square == square:
                                        to_row = 7 - chess.square_rank(move.to_square)
                                        to_col = chess.square_file(move.to_square)
                                        game.valid_moves.append((to_row, to_col))
                                print(f"Selected piece at {row},{col} ({chess.square_name(square)}), valid moves: {[(r, c, chess.square_name(chess.square(c, 7 - r))) for r, c in game.valid_moves]}")
                        else:
                            to_row, to_col = row, col
                            to_square = chess.square(to_col, 7 - to_row)
                            from_row, from_col = game.selected
                            from_square = chess.square(from_col, 7 - from_row)
                            print(f"Destination square: ({to_row}, {to_col}) -> chess square {to_square} ({chess.square_name(to_square)})")
                            dest_piece = game.board.chess_board.piece_at(to_square)
                            print(f"Destination square {chess.square_name(to_square)} contains: {dest_piece if dest_piece else 'None'}")
                            if dest_piece and dest_piece.color != (chess.WHITE if self.color == 'w' else chess.BLACK):
                                game.captures[self.color] += 1
                            legal_moves = game.board.get_all_moves(self.color)
                            selected_move = None
                            for move in legal_moves:
                                if move.from_square == from_square and move.to_square == to_square:
                                    selected_move = move
                                    if (game.board.chess_board.piece_at(from_square).piece_type == chess.PAWN and
                                        (to_row == 0 or to_row == 7) and not move.promotion):
                                        selected_move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
                                    break
                            if selected_move:
                                print(f"Move accepted: {selected_move.uci()}")
                                return selected_move
                            else:
                                print(f"Move rejected: No legal move found from {chess.square_name(from_square)} to {chess.square_name(to_square)}")
                            game.selected = None
                            game.valid_moves = []
                            print("Deselected piece")
            game.draw()
            await asyncio.sleep(0)

class AIPlayer(Player):
    def __init__(self, color, depth=3):
        super().__init__(color)
        self.depth = depth
        self.evaluation = Evaluation()
        self.last_eval = 0

    async def get_move(self, game):
        if not game.running:
            print("Game is over, AI cannot make moves")
            return None
        game.status = "AI is thinking..."
        game.draw()
        await asyncio.sleep(0.1)
        best_move, eval_score = self.minimax(game.board, self.depth, float("-inf"), float("inf"), True)
        self.last_eval = eval_score
        game.status = ""
        if best_move is None:
            print("AI: No valid move found - likely game over")
            if game.check_game_over():
                return None
            else:
                raise Exception("AI failed to find a move despite game not being over")
        print(f"AI move calculated: {best_move.uci()}")
        dest_piece = game.board.chess_board.piece_at(best_move.to_square)
        if dest_piece and dest_piece.color != (chess.WHITE if self.color == 'w' else chess.BLACK):
            game.captures[self.color] += 1
        return best_move

    def minimax(self, board, depth, alpha, beta, maximizing):
        if depth == 0:
            score, _ = self.evaluation.evaluate(board, self.color)
            return None, score

        moves = board.get_all_moves(self.color if maximizing else ('w' if self.color == 'b' else 'b'))
        if not moves:
            if board.chess_board.is_checkmate():
                return None, float("-inf") if maximizing else float("inf")
            return None, 0

        best_move = None
        if maximizing:
            max_eval = float("-inf")
            for move in moves:
                board.make_move(move)
                _, eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.undo_move()
                if math.isnan(eval):
                    eval = float("-inf")
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return best_move, max_eval
        else:
            min_eval = float("inf")
            for move in moves:
                board.make_move(move)
                _, eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.undo_move()
                if math.isnan(eval):
                    eval = float("inf")
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return best_move, min_eval

class ChessGame:
    def __init__(self):
        self.board = Board()
        self.players = [HumanPlayer("w"), AIPlayer("b")]
        self.current_player = 0
        self.selected = None
        self.valid_moves = []
        self.move_history = []
        self.status = ""
        self.running = True
        self.total_moves = 0
        self.captures = {"w": 0, "b": 0}

    def check_game_over(self):
        if self.board.chess_board.is_checkmate():
            color = self.players[self.current_player].color
            self.status = f"Checkmate! {'Black' if color == 'w' else 'White'} wins!"
            print(f"Game over: {self.status}")
            self.running = False
            return True
        if self.board.chess_board.is_stalemate():
            self.status = "Stalemate!"
            print(f"Game over: {self.status}")
            self.running = False
            return True
        print("No game-over condition detected")
        return False

    def draw(self, win_message=None):
        for row in range(8):
            for col in range(8):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                square = chess.square(col, 7 - row)
                piece = self.board.chess_board.piece_at(square)
                if piece:
                    color = 'w' if piece.color == chess.WHITE else 'b'
                    symbol = piece.symbol().upper()
                    piece_img = self.board.piece_map.get((color, symbol))
                    if piece_img:
                        screen.blit(piece_img.image, (col * SQUARE_SIZE, row * SQUARE_SIZE))

        if self.selected:
            row, col = self.selected
            highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            highlight_surface.fill(HIGHLIGHT)
            screen.blit(highlight_surface, (col * SQUARE_SIZE, row * SQUARE_SIZE))
            for move in self.valid_moves:
                screen.blit(highlight_surface, (move[1] * SQUARE_SIZE, move[0] * SQUARE_SIZE))

        pygame.draw.rect(screen, BLACK, (BOARD_SIZE, 0, HISTORY_WIDTH, HEIGHT))
        for i, move in enumerate(self.move_history[-20:]):
            text = font.render(move.uci(), True, WHITE)
            screen.blit(text, (BOARD_SIZE + 10, 10 + i * 20))

        status_text = font.render(self.status, True, BLACK)
        pygame.draw.rect(screen, WHITE, (0, 0, BOARD_SIZE, 20))
        screen.blit(status_text, (10, 0))

        if win_message:
            overlay = pygame.Surface((BOARD_SIZE, HEIGHT), pygame.SRCALPHA)
            overlay.fill(WIN_OVERLAY)
            screen.blit(overlay, (0, 0))
            win_text = win_font.render(win_message, True, WHITE)
            win_rect = win_text.get_rect(center=(BOARD_SIZE // 2, HEIGHT // 2))
            screen.blit(win_text, win_rect)

        pygame.display.flip()

    async def play(self):
        self.draw()
        while self.running:
            try:
                if self.check_game_over():
                    win_message = self.status
                    self.draw(win_message=win_message)
                    await asyncio.sleep(10)
                    final_eval, material = self.players[1].evaluation.evaluate(self.board, 'b')
                    print("\n=== Game Metrics ===")
                    print(f"Total Moves: {self.total_moves}")
                    print(f"Captures - White: {self.captures['w']}, Black: {self.captures['b']}")
                    print(f"Final Alpha-Beta Score (Black's perspective): {self.players[1].last_eval}")
                    print(f"Final Board Evaluation (Black's perspective): {final_eval}")
                    print(f"Material - White: {material['w']}, Black: {material['b']}")
                    break

                print(f"Turn: Player {self.current_player} ({self.players[self.current_player].color})")
                move = await self.players[self.current_player].get_move(self)
                if move is None:
                    print("Game loop: Quit signal received")
                    self.running = False
                    break

                self.board.make_move(move)
                self.move_history.append(move)
                self.selected = None
                self.valid_moves = []
                self.current_player = (self.current_player + 1) % 2
                self.total_moves += 1
                self.draw()
                print(f"Move completed, switching to player {self.current_player}")

                await asyncio.sleep(0)

            except Exception as e:
                print(f"Error in game loop: {e}")
                self.running = False
                break

        pygame.quit()

async def main():
    try:
        game = ChessGame()
        await game.play()
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        pygame.quit()

if platform.system() == "Emscripten":
    import js
    import pyodide

    async def run_game():
        try:
            await main()
        except Exception as e:
            print(f"Error in run_game: {e}")

    pyodide.create_proxy(run_game())()
else:
    if __name__ == "__main__":
        asyncio.run(main())