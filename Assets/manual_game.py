import pygame
import os
import platform
import asyncio
from abc import ABC, abstractmethod
import math

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
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (255, 255, 0, 100)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Game: Human vs AI")
font = pygame.font.Font(None, 24)

class Move:
    def __init__(self, from_pos, to_pos, piece, move_type="normal", captured=None, promotion=None):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.piece = piece
        self.move_type = move_type
        self.captured = captured
        self.promotion = promotion

    def to_notation(self):
        files = "abcdefgh"
        ranks = "87654321"
        from_notation = f"{files[self.from_pos[1]]}{ranks[self.from_pos[0]]}"
        to_notation = f"{files[self.to_pos[1]]}{ranks[self.to_pos[0]]}"
        if self.move_type == "castling":
            return "O-O" if self.to_pos[1] > self.from_pos[1] else "O-O-O"
        return f"{self.piece.symbol}{from_notation}-{to_notation}"

class Piece(ABC):
    def __init__(self, color):
        self.color = color
        self.has_moved = False
        try:
            self.image = pygame.image.load(os.path.join("Assets", f"{self.color}{self.symbol}.png"))
            self.image = pygame.transform.scale(self.image, (SQUARE_SIZE, SQUARE_SIZE))
        except pygame.error as e:
            print(f"Error loading image for {self.color}{self.symbol}: {e}")
            self.image = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
            self.image.fill(BLACK if self.color == "b" else WHITE)

    @abstractmethod
    def get_valid_moves(self, board, pos, checking=False):
        pass

class Pawn(Piece):
    symbol = "P"
    value = 1

    def get_valid_moves(self, board, pos, checking=False):
        moves = []
        captures = []
        direction = -1 if self.color == "w" else 1
        start_row = 6 if self.color == "w" else 1

        new_row = pos[0] + direction
        if 0 <= new_row < 8 and board.squares[new_row][pos[1]] is None:
            moves.append((new_row, pos[1]))
            if pos[0] == start_row:
                new_row += direction
                if board.squares[new_row][pos[1]] is None:
                    moves.append((new_row, pos[1]))

        for dc in [-1, 1]:
            new_col = pos[1] + dc
            if 0 <= new_col < 8:
                new_row = pos[0] + direction
                if 0 <= new_row < 8:
                    piece = board.squares[new_row][new_col]
                    if piece and piece.color != self.color:
                        captures.append((new_row, new_col))
                if pos[0] == (4 if self.color == "w" else 3):
                    side_piece = board.squares[pos[0]][new_col]
                    last_move = board.last_move
                    if (side_piece and side_piece.symbol == "P" and side_piece.color != self.color and
                        last_move and last_move.piece == side_piece and
                        last_move.from_pos[0] == pos[0] + 2 * direction and last_move.to_pos[0] == pos[0]):
                        captures.append((pos[0] + direction, new_col))

        return moves, captures

class Knight(Piece):
    symbol = "N"
    value = 3

    def get_valid_moves(self, board, pos, checking=False):
        moves = []
        captures = []
        directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for dr, dc in directions:
            new_row, new_col = pos[0] + dr, pos[1] + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                piece = board.squares[new_row][new_col]
                if piece is None:
                    moves.append((new_row, new_col))
                elif piece.color != self.color:
                    captures.append((new_row, new_col))
        return moves, captures

class Bishop(Piece):
    symbol = "B"
    value = 3

    def get_valid_moves(self, board, pos, checking=False):
        moves = []
        captures = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            new_row, new_col = pos[0] + dr, pos[1] + dc
            while 0 <= new_row < 8 and 0 <= new_col < 8:
                piece = board.squares[new_row][new_col]
                if piece is None:
                    moves.append((new_row, new_col))
                elif piece.color != self.color:
                    captures.append((new_row, new_col))
                    break
                else:
                    break
                new_row += dr
                new_col += dc
        return moves, captures

class Rook(Piece):
    symbol = "R"
    value = 5

    def get_valid_moves(self, board, pos, checking=False):
        moves = []
        captures = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = pos[0] + dr, pos[1] + dc
            while 0 <= new_row < 8 and 0 <= new_col < 8:
                piece = board.squares[new_row][new_col]
                if piece is None:
                    moves.append((new_row, new_col))
                elif piece.color != self.color:
                    captures.append((new_row, new_col))
                    break
                else:
                    break
                new_row += dr
                new_col += dc
        return moves, captures

class Queen(Piece):
    symbol = "Q"
    value = 9

    def get_valid_moves(self, board, pos, checking=False):
        moves = []
        captures = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            new_row, new_col = pos[0] + dr, pos[1] + dc
            while 0 <= new_row < 8 and 0 <= new_col < 8:
                piece = board.squares[new_row][new_col]
                if piece is None:
                    moves.append((new_row, new_col))
                elif piece.color != self.color:
                    captures.append((new_row, new_col))
                    break
                else:
                    break
                new_row += dr
                new_col += dc
        return moves, captures

class King(Piece):
    symbol = "K"
    value = 1000  # Changed from float("inf") to a large finite value to avoid nan

    def get_valid_moves(self, board, pos, checking=False):
        moves = []
        captures = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in directions:
            new_row, new_col = pos[0] + dr, pos[1] + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                piece = board.squares[new_row][new_col]
                if piece is None:
                    moves.append((new_row, new_col))
                elif piece.color != self.color:
                    captures.append((new_row, new_col))

        if not checking and not self.has_moved:
            if (board.squares[pos[0]][5] is None and board.squares[pos[0]][6] is None and
                isinstance(board.squares[pos[0]][7], Rook) and not board.squares[pos[0]][7].has_moved):
                board.squares[pos[0]][4] = None
                board.squares[pos[0]][5] = self
                attacked_5 = board.is_square_attacked((pos[0], 5), self.color, checking=True)
                board.squares[pos[0]][5] = None
                board.squares[pos[0]][6] = self
                attacked_6 = board.is_square_attacked((pos[0], 6), self.color, checking=True)
                board.squares[pos[0]][6] = None
                board.squares[pos[0]][4] = self
                if not attacked_5 and not attacked_6:
                    moves.append((pos[0], 6))
            if (board.squares[pos[0]][3] is None and board.squares[pos[0]][2] is None and board.squares[pos[0]][1] is None and
                isinstance(board.squares[pos[0]][0], Rook) and not board.squares[pos[0]][0].has_moved):
                board.squares[pos[0]][4] = None
                board.squares[pos[0]][3] = self
                attacked_3 = board.is_square_attacked((pos[0], 3), self.color, checking=True)
                board.squares[pos[0]][3] = None
                board.squares[pos[0]][2] = self
                attacked_2 = board.is_square_attacked((pos[0], 2), self.color, checking=True)
                board.squares[pos[0]][2] = None
                board.squares[pos[0]][4] = self
                if not attacked_3 and not attacked_2:
                    moves.append((pos[0], 2))

        return moves, captures

class Board:
    def __init__(self):
        self.squares = [[None for _ in range(8)] for _ in range(8)]
        self.last_move = None
        self.setup_board()

    def setup_board(self):
        for col in range(8):
            self.squares[1][col] = Pawn("b")
            self.squares[6][col] = Pawn("w")
        self.squares[0][0] = Rook("b")
        self.squares[0][7] = Rook("b")
        self.squares[7][0] = Rook("w")
        self.squares[7][7] = Rook("w")
        self.squares[0][1] = Knight("b")
        self.squares[0][6] = Knight("b")
        self.squares[7][1] = Knight("w")
        self.squares[7][6] = Knight("w")
        self.squares[0][2] = Bishop("b")
        self.squares[0][5] = Bishop("b")
        self.squares[7][2] = Bishop("w")
        self.squares[7][5] = Bishop("w")
        self.squares[0][3] = Queen("b")
        self.squares[7][3] = Queen("w")
        self.squares[0][4] = King("b")
        self.squares[7][4] = King("w")

    def is_in_check(self, color):
        king_pos = None
        for r in range(8):
            for c in range(8):
                piece = self.squares[r][c]
                if piece and piece.symbol == "K" and piece.color == color:
                    king_pos = (r, c)
                    break
        if not king_pos:
            return False
        return self.is_square_attacked(king_pos, color)

    def is_square_attacked(self, pos, color, checking=False):
        for r in range(8):
            for c in range(8):
                piece = self.squares[r][c]
                if piece and piece.color != color:
                    moves, captures = piece.get_valid_moves(self, (r, c), checking=checking)
                    if pos in moves or pos in captures:
                        return True
        return False

    def get_all_moves(self, color):
        moves = []
        for r in range(8):
            for c in range(8):
                piece = self.squares[r][c]
                if piece and piece.color == color:
                    piece_moves, piece_captures = piece.get_valid_moves(self, (r, c))
                    for to_pos in piece_moves + piece_captures:
                        move = Move((r, c), to_pos, piece)
                        if piece.symbol == "K" and abs(to_pos[1] - (r, c)[1]) == 2:
                            move.move_type = "castling"
                        moves.append(move)
        print(f"Generated {len(moves)} moves for {color}")
        return moves

    def make_move(self, move):
        from_pos, to_pos = move.from_pos, move.to_pos
        piece = self.squares[from_pos[0]][from_pos[1]]
        captured = self.squares[to_pos[0]][to_pos[1]]
        move.captured = captured

        if move.move_type == "castling":
            rook_col = 7 if to_pos[1] > from_pos[1] else 0
            rook_new_col = 5 if to_pos[1] > from_pos[1] else 3
            self.squares[from_pos[0]][rook_new_col] = self.squares[from_pos[0]][rook_col]
            self.squares[from_pos[0]][rook_col] = None
            self.squares[from_pos[0]][rook_new_col].has_moved = True

        if piece.symbol == "P" and to_pos[1] != from_pos[1] and captured is None:
            captured_row = from_pos[0]
            self.squares[captured_row][to_pos[1]] = None
            move.captured = Pawn("b" if piece.color == "w" else "w")

        if piece.symbol == "P" and to_pos[0] in (0, 7):
            piece = Queen(piece.color)
            move.promotion = "Q"

        self.squares[to_pos[0]][to_pos[1]] = piece
        self.squares[from_pos[0]][from_pos[1]] = None
        piece.has_moved = True
        self.last_move = move

    def undo_move(self, move):
        from_pos, to_pos = move.from_pos, move.to_pos
        piece = self.squares[to_pos[0]][to_pos[1]]
        self.squares[from_pos[0]][from_pos[1]] = piece
        self.squares[to_pos[0]][to_pos[1]] = move.captured

        if move.move_type == "castling":
            rook_col = 7 if to_pos[1] > from_pos[1] else 0
            rook_new_col = 5 if to_pos[1] > from_pos[1] else 3
            self.squares[from_pos[0]][rook_col] = self.squares[from_pos[0]][rook_new_col]
            self.squares[from_pos[0]][rook_new_col] = None
            self.squares[from_pos[0]][rook_col].has_moved = False

        if piece.symbol == "P" and move.captured and move.captured.symbol == "P" and to_pos[1] != from_pos[1] and self.squares[to_pos[0]][to_pos[1]] is None:
            captured_row = from_pos[0]
            self.squares[captured_row][to_pos[1]] = move.captured

        if move.promotion:
            piece = Pawn(piece.color)

        piece.has_moved = False
        self.last_move = None

class Evaluation:
    def evaluate(self, board, color):
        piece_counts = {"w": 0, "b": 0}
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]

        for r in range(8):
            for c in range(8):
                piece = board.squares[r][c]
                if piece:
                    # Skip the king for material count to avoid inf - inf
                    if piece.symbol != "K":
                        piece_counts[piece.color] += piece.value
                    # Positional bonuses still apply to all pieces
                    if (r, c) in center_squares:
                        piece_counts[piece.color] += 0.5
                    if piece.symbol == "K":
                        if board.is_square_attacked((r, c), piece.color):
                            piece_counts[piece.color] -= 1
                        if piece.has_moved:
                            piece_counts[piece.color] += 0.5

        score = piece_counts[color] - piece_counts["b" if color == "w" else "w"]
        print(f"Evaluation for {color}: piece_counts={piece_counts}, score={score}")
        if math.isnan(score):
            raise ValueError("Evaluation returned nan")
        return score

class Player(ABC):
    def __init__(self, color):
        self.color = color

    @abstractmethod
    async def get_move(self, game):
        pass

class HumanPlayer(Player):
    async def get_move(self, game):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Quit event detected in HumanPlayer")
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    row, col = pos[1] // SQUARE_SIZE, pos[0] // SQUARE_SIZE
                    if 0 <= row < 8 and 0 <= col < 8:
                        if not game.selected:
                            piece = game.board.squares[row][col]
                            if piece and piece.color == self.color:
                                game.selected = (row, col)
                                game.valid_moves, game.valid_captures = piece.get_valid_moves(game.board, (row, col))
                                print(f"Selected piece at {row},{col}")
                        else:
                            to_pos = (row, col)
                            if to_pos in game.valid_moves + game.valid_captures:
                                move = Move(game.selected, to_pos, game.board.squares[game.selected[0]][game.selected[1]])
                                if move.piece.symbol == "K" and abs(to_pos[1] - game.selected[1]) == 2:
                                    move.move_type = "castling"
                                if game.is_legal_move(move):
                                    print(f"Valid move made from {game.selected} to {to_pos}")
                                    return move
                            game.selected = None
                            game.valid_moves = []
                            game.valid_captures = []
                            print("Deselected piece")
            game.draw()
            await asyncio.sleep(0)

class AIPlayer(Player):
    def __init__(self, color, depth=3):
        super().__init__(color)
        self.depth = depth
        self.evaluation = Evaluation()

    async def get_move(self, game):
        game.status = "AI is thinking..."
        game.draw()
        await asyncio.sleep(0.5)  # Small delay to ensure UI updates
        best_move, _ = self.minimax(game.board, self.depth, float("-inf"), float("inf"), True)
        game.status = ""
        if best_move is None:
            print("AI: No valid move found - likely game over")
            if game.check_game_over():
                return None  # Game over condition
            else:
                raise Exception("AI failed to find a move despite game not being over")
        print(f"AI move calculated: {best_move.to_notation()}")
        return best_move

    def minimax(self, board, depth, alpha, beta, maximizing):
        if depth == 0:
            score = self.evaluation.evaluate(board, self.color)
            return None, score

        moves = board.get_all_moves(self.color if maximizing else ("w" if self.color == "b" else "b"))
        if not moves:
            if board.is_in_check(self.color if maximizing else ("w" if self.color == "b" else "b")):
                return None, float("-inf") if maximizing else float("inf")
            return None, 0

        best_move = None
        if maximizing:
            max_eval = float("-inf")
            for move in moves:
                board.make_move(move)
                _, eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.undo_move(move)
                print(f"Maximizing: Evaluated move {move.to_notation()} with score {eval}")
                if math.isnan(eval):
                    eval = float("-inf")  # Safeguard against nan
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    print("Pruning: beta <= alpha")
                    break
            print(f"Maximizing: Best move {best_move.to_notation() if best_move else 'None'} with score {max_eval}")
            return best_move, max_eval
        else:
            min_eval = float("inf")
            for move in moves:
                board.make_move(move)
                _, eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.undo_move(move)
                print(f"Minimizing: Evaluated move {move.to_notation()} with score {eval}")
                if math.isnan(eval):
                    eval = float("inf")  # Safeguard against nan
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    print("Pruning: beta <= alpha")
                    break
            print(f"Minimizing: Best move {best_move.to_notation() if best_move else 'None'} with score {min_eval}")
            return best_move, min_eval

class ChessGame:
    def __init__(self):
        self.board = Board()
        self.players = [HumanPlayer("w"), AIPlayer("b")]
        self.current_player = 0
        self.selected = None
        self.valid_moves = []
        self.valid_captures = []
        self.move_history = []
        self.status = ""
        self.running = True

    def is_legal_move(self, move):
        self.board.make_move(move)
        in_check = self.board.is_in_check(self.players[self.current_player].color)
        self.board.undo_move(move)
        return not in_check

    def check_game_over(self):
        color = self.players[self.current_player].color
        moves = self.board.get_all_moves(color)
        if not moves:
            if self.board.is_in_check(color):
                self.status = f"Checkmate! {'White' if color == 'b' else 'Black'} wins!"
            else:
                self.status = "Stalemate!"
            return True
        return False

    def draw(self):
        for row in range(8):
            for col in range(8):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                piece = self.board.squares[row][col]
                if piece:
                    screen.blit(piece.image, (col * SQUARE_SIZE, row * SQUARE_SIZE))

        if self.selected:
            row, col = self.selected
            highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            highlight_surface.fill(HIGHLIGHT)
            screen.blit(highlight_surface, (col * SQUARE_SIZE, row * SQUARE_SIZE))
            for move in self.valid_moves + self.valid_captures:
                screen.blit(highlight_surface, (move[1] * SQUARE_SIZE, move[0] * SQUARE_SIZE))

        pygame.draw.rect(screen, BLACK, (BOARD_SIZE, 0, HISTORY_WIDTH, HEIGHT))
        for i, move in enumerate(self.move_history[-20:]):
            text = font.render(move.to_notation(), True, WHITE)
            screen.blit(text, (BOARD_SIZE + 10, 10 + i * 20))

        status_text = font.render(self.status, True, BLACK)
        pygame.draw.rect(screen, WHITE, (0, 0, BOARD_SIZE, 20))
        screen.blit(status_text, (10, 0))

        pygame.display.flip()

    async def play(self):
        self.draw()
        while self.running:
            try:
                if self.check_game_over():
                    self.draw()
                    await asyncio.sleep(5)
                    break

                print(f"Turn: Player {self.current_player} ({self.players[self.current_player].color})")
                move = await self.players[self.current_player].get_move(self)
                if move is None and isinstance(self.players[self.current_player], HumanPlayer):
                    print("Game loop: Quit signal received from HumanPlayer")
                    self.running = False
                    break
                elif move is None and isinstance(self.players[self.current_player], AIPlayer):
                    if self.check_game_over():
                        self.draw()
                        await asyncio.sleep(5)
                        break
                    else:
                        print("AI returned None but game not over - this should not happen")
                        self.running = False
                        break

                self.board.make_move(move)
                self.move_history.append(move)
                self.selected = None
                self.valid_moves = []
                self.valid_captures = []
                self.current_player = (self.current_player + 1) % 2
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