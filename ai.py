import numpy as np
import math
from copy import deepcopy

class AIPlayer:
    def __init__(self, player=2, depth=3):
        self.player = player  # AI玩家编号(2为白棋)
        self.depth = depth    # 搜索深度
        self.weights = [      # 棋型评分权重
            [0, 0, 1, 10, 100, 10000],       # 连1, 连2, 连3, 连4, 连5
            [0, 0, 1, 10, 100, 10000],       # 对手的连子
        ]
    
    def get_move(self, game):
        """获取AI的最佳落子位置"""
        if len(game.get_empty_positions()) == game.size * game.size:
            # 开局中心落子
            return game.size // 2, game.size // 2
        
        # 使用Alpha-Beta剪枝搜索最佳落子
        _, move = self.alphabeta(game, self.depth, -math.inf, math.inf, True)
        return move
    
    def alphabeta(self, game, depth, alpha, beta, maximizing_player):
        """Alpha-Beta剪枝算法"""
        if depth == 0 or game.game_over:
            return self.evaluate(game), None
        
        empty_positions = game.get_empty_positions()
        
        if maximizing_player:
            max_eval = -math.inf
            best_move = None
            
            for move in empty_positions:
                x, y = move
                game_copy = deepcopy(game)
                game_copy.make_move(x, y)
                
                evaluation, _ = self.alphabeta(game_copy, depth-1, alpha, beta, False)
                
                if evaluation > max_eval:
                    max_eval = evaluation
                    best_move = move
                
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        else:
            min_eval = math.inf
            best_move = None
            
            for move in empty_positions:
                x, y = move
                game_copy = deepcopy(game)
                game_copy.make_move(x, y)
                
                evaluation, _ = self.alphabeta(game_copy, depth-1, alpha, beta, True)
                
                if evaluation < min_eval:
                    min_eval = evaluation
                    best_move = move
                
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            
            return min_eval, best_move
    
    def evaluate(self, game):
        """评估当前棋盘状态"""
        score = 0
        
        # 检查所有可能的五连位置
        for y in range(game.size):
            for x in range(game.size):
                if game.board[y][x] != 0:
                    # 计算每个棋子四个方向的得分
                    for dy, dx in [(0,1), (1,0), (1,1), (1,-1)]:
                        score += self.evaluate_direction(game, x, y, dx, dy)
        
        return score if self.player == 1 else -score
    
    def evaluate_direction(self, game, x, y, dx, dy):
        """评估特定方向的棋型"""
        player = game.board[y][x]
        opponent = 3 - player
        score = 0
        count = 1       # 连续棋子数
        open_ends = 0   # 开放端数量(0,1,2)
        
        # 检查正向
        nx, ny = x + dx, y + dy
        while 0 <= nx < game.size and 0 <= ny < game.size:
            if game.board[ny][nx] == player:
                count += 1
            elif game.board[ny][nx] == 0:
                open_ends += 1
                break
            else:
                break
            nx += dx
            ny += dy
        
        # 检查反向
        nx, ny = x - dx, y - dy
        while 0 <= nx < game.size and 0 <= ny < game.size:
            if game.board[ny][nx] == player:
                count += 1
            elif game.board[ny][nx] == 0:
                open_ends += 1
                break
            else:
                break
            nx -= dx
            ny -= dy
        
        # 根据连子数和开放端计算得分
        if count >= 5:
            return self.weights[0][5]  # 五连
        
        if open_ends == 0:
            return 0  # 封闭的连子没有价值
        
        # 为接近中心的位置提供更高的得分
        center_distance = abs(x - game.size // 2) + abs(y - game.size // 2)
        center_bonus = max(0, game.size // 2 - center_distance)  # 距离中心越近，得分越高
        
        return self.weights[0][count] * open_ends + center_bonus
