import numpy as np


class GomokuGame:
    def __init__(self, size=15):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # 1:黑, 2:白
        self.game_over = False
        self.winner = None
    
    def make_move(self, x, y, player=None):
        """执行落子并检查胜负。支持指定玩家（用于AI搜索）"""
        if self.game_over or self.board[y][x] != 0:
            return False

        if player is None:
            player = self.current_player

        self.board[y][x] = player

        if self.check_win(x, y, player):
            self.game_over = True
            self.winner = player
            return True

        if player == self.current_player:
            self.current_player = 3 - self.current_player  # 只在真实对局中切换玩家

        return True

    
    def check_win(self, x, y, player):
        """检查五子连珠，支持指定玩家"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            # 正向
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.size and 0 <= ny < self.size and self.board[ny][nx] == player:
                count += 1
                nx += dx
                ny += dy
            # 反向
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.size and 0 <= ny < self.size and self.board[ny][nx] == player:
                count += 1
                nx -= dx
                ny -= dy
            if count >= 5:
                return True
        return False

    
    def undo_move(self, x, y):
        """撤销指定位置的落子"""
        if self.board[y][x] == 0:
            return False  # 该位置原本就是空的，无需撤销
        
        # 恢复游戏状态
        self.board[y][x] = 0
        self.current_player = 3 - self.current_player  # 切换回上一个玩家
        self.game_over = False
        self.winner = None
        return True
    
    def get_empty_positions(self):
        """获取所有空位置"""
        return [(x, y) for x in range(self.size) 
                for y in range(self.size) if self.board[y][x] == 0]
    
    def reset(self):
        """重置游戏状态"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None

