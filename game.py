import numpy as np

class GomokuGame:
    def __init__(self, size=15):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # 1:黑, 2:白
        self.game_over = False
        self.winner = None
    
    def make_move(self, x, y):
        """执行落子并检查胜负"""
        if self.game_over or self.board[y][x] != 0:
            return False
        
        self.board[y][x] = self.current_player
        
        if self.check_win(x, y):
            self.game_over = True
            self.winner = self.current_player
            return True
        
        self.current_player = 3 - self.current_player  # 切换玩家
        return True
    
    def check_win(self, x, y):
        """检查五子连珠"""
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for dx, dy in directions:
            count = 1
            # 正向检查
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.size and 0 <= ny < self.size and self.board[ny][nx] == self.current_player:
                count += 1
                nx += dx
                ny += dy
            # 反向检查
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.size and 0 <= ny < self.size and self.board[ny][nx] == self.current_player:
                count += 1
                nx -= dx
                ny -= dy
            if count >= 5:
                return True
        return False
    
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
