import numpy as np
import math
from copy import deepcopy

class AIPlayer:
    def __init__(self, player=2, depth=3):
        self.player = player  # AI(2为白棋)
        self.depth = depth    # 搜索深度
        self.weights = [      # 棋型评分权重
            [0, 0, 1, 10, 100, 10000],       # 不连, 连1, 连2, 连3, 连4, 连5
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
    
    def get_candidate_positions(self, game, distance=2):# 候选位置过滤,只考虑已有棋子周围 2 格范围内的空位
        candidates = set()
        for y in range(game.size):
            for x in range(game.size):
                if game.board[y][x] != 0:
                    for dy in range(-distance, distance + 1):
                        for dx in range(-distance, distance + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < game.size and 0 <= ny < game.size:
                                if game.board[ny][nx] == 0:
                                    candidates.add((nx, ny))
        if not candidates:
            center = game.size // 2
            return [(center, center)]
        return list(candidates)

    def alphabeta(self, game, depth, alpha, beta, maximizing_player):
        """Alpha-Beta剪枝算法"""
        if depth == 0 or game.game_over:
            return self.evaluate(game), None
        
        # empty_positions = game.get_empty_positions()
        empty_positions = self.get_candidate_positions(game)
        
        if maximizing_player:
            max_eval = -math.inf
            best_move = None
            # 副本模拟落子
            for move in empty_positions:
                x, y = move
                # game_copy = deepcopy(game)
                player = self.player if maximizing_player else 3 - self.player
                game.make_move(x, y, player)# 直接落子

                # game.make_move(x, y)  
                # 递归评估对手回合
                evaluation, _ = self.alphabeta(game, depth-1, alpha, beta, False)
                
                game.undo_move(x, y)  # 撤销落子

                if evaluation > max_eval:
                    max_eval = evaluation
                    best_move = move

                # Alpha-Beta 剪枝
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        else:
            min_eval = math.inf
            best_move = None
            
            for move in empty_positions:
                x, y = move

                player = self.player if maximizing_player else 3 - self.player
                game.make_move(x, y, player)

                # game.make_move(x, y)  # 直接落子
                
                evaluation, _ = self.alphabeta(game, depth-1, alpha, beta, True)
                
                game.undo_move(x, y)  # 撤销落子
                if evaluation < min_eval:
                    min_eval = evaluation
                    best_move = move
                
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            
            return min_eval, best_move
    
    # def evaluate(self, game):
    #     """评估当前棋盘状态"""
    #     score = 0
        
    #     # 检查所有可能的五连位置
    #     for y in range(game.size):
    #         for x in range(game.size):
    #             if game.board[y][x] != 0:
    #                 # 计算每个棋子四个方向的得分
    #                 for dy, dx in [(0,1), (1,0), (1,1), (1,-1)]:
    #                     score += self.evaluate_direction(game, x, y, dx, dy)
        
    #     return score if self.player == 1 else -score

    def evaluate(self, game):# 分离 AI 和对手得分
        """敌我分开评分，返回总优势值"""
        my_score = 0
        opp_score = 0
        for y in range(game.size):
            for x in range(game.size):
                current = game.board[y][x]
                if current == 0:
                    continue
                for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    score = self.evaluate_direction(game, x, y, dx, dy, current)
                    if current == self.player:
                        my_score += score
                    else:
                        opp_score += score
        return my_score - opp_score

    
    # def evaluate_direction(self, game, x, y, dx, dy):
    #     """评估特定方向的棋型"""
    #     player = game.board[y][x] # 当前棋子所属玩家（1或2）
    #     opponent = 3 - player
    #     score = 0
    #     count = 1       # 连续棋子数
    #     open_ends = 0   # 开放端数量(0,1,2)（0=封闭，1=半开放，2=全开放）
        
    #     # 检查正向
    #     nx, ny = x + dx, y + dy
    #     while 0 <= nx < game.size and 0 <= ny < game.size:
    #         if game.board[ny][nx] == player:
    #             count += 1
    #         elif game.board[ny][nx] == 0:
    #             open_ends += 1
    #             break
    #         else:
    #             break
    #         nx += dx
    #         ny += dy
        
    #     # 检查反向
    #     nx, ny = x - dx, y - dy
    #     while 0 <= nx < game.size and 0 <= ny < game.size:
    #         if game.board[ny][nx] == player:
    #             count += 1
    #         elif game.board[ny][nx] == 0:
    #             open_ends += 1
    #             break
    #         else:
    #             break
    #         nx -= dx
    #         ny -= dy
        
    #     # 根据连子数和开放端计算得分
    #     if count >= 5:
    #         return self.weights[0][5]  # 五连
        
    #     if open_ends == 0:
    #         return 0  # 封闭的连子没有价值
        
    #     # 为接近中心的位置提供更高的得分
    #     center_distance = abs(x - game.size // 2) + abs(y - game.size // 2)
    #     center_bonus = (game.size // 2 - center_distance) * 10  # 距离中心越近，得分越高
        
    #     return self.weights[0][count] * open_ends + center_bonus
    
    def evaluate_direction(self, game, x, y, dx, dy, player):
        """评估某方向的棋型（活三、冲四等）"""
        total = 1
        block = 0
        empty = 0
        max_len = 5  # 最多看5个

        # 正方向
        i = 1
        while i < max_len:
            nx = x + dx * i
            ny = y + dy * i
            if 0 <= nx < game.size and 0 <= ny < game.size:
                if game.board[ny][nx] == player:
                    total += 1
                elif game.board[ny][nx] == 0:
                    empty += 1
                    break
                else:
                    block += 1
                    break
            else:
                block += 1
                break
            i += 1

        # 反方向
        i = 1
        while i < max_len:
            nx = x - dx * i
            ny = y - dy * i
            if 0 <= nx < game.size and 0 <= ny < game.size:
                if game.board[ny][nx] == player:
                    total += 1
                elif game.board[ny][nx] == 0:
                    empty += 1
                    break
                else:
                    block += 1
                    break
            else:
                block += 1
                break
            i += 1

        # 根据连子长度和封堵情况评分
        if total >= 5:
            return 100000
        elif total == 4:
            if block == 0:
                return 10000  # 活四
            elif block == 1:
                return 1000   # 冲四
        elif total == 3:
            if block == 0:
                return 500   # 活三
            elif block == 1:
                return 100   # 眠三
        elif total == 2:
            if block == 0:
                return 50    # 活二
            elif block == 1:
                return 10    # 眠二
        return 1
