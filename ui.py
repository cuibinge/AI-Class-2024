import tkinter as tk
from tkinter import messagebox
import time

class GomokuGUI:
    def __init__(self, game, ai_player):
        self.game = game
        self.ai_player = ai_player
        self.setup_constants()
        self.setup_ui()
    
    def setup_constants(self):
        self.size = 15
        self.cell_size = 40
        self.margin = 30
        self.radius = 15
    
    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("五子棋")
        
        # 计算画布尺寸
        canvas_size = self.margin * 2 + (self.size - 1) * self.cell_size
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg='#DCB35C')
        self.canvas.pack()
        
        self.draw_board()
        self.canvas.bind("<Button-1>", self.handle_click)
        
        self.status_var = tk.StringVar()
        self.update_status()
        tk.Label(self.root, textvariable=self.status_var, font=('Arial', 14)).pack()
        tk.Button(self.root, text="重新开始", command=self.reset_game).pack()
    
    def run(self):
        self.root.mainloop()
    
    def draw_board(self):
        # 绘制棋盘网格
        for i in range(self.size):
            self.canvas.create_line(
                self.margin, self.margin + i * self.cell_size,
                self.margin + (self.size - 1) * self.cell_size, self.margin + i * self.cell_size,
                width=2
            )
            self.canvas.create_line(
                self.margin + i * self.cell_size, self.margin,
                self.margin + i * self.cell_size, self.margin + (self.size - 1) * self.cell_size,
                width=2
            )
        
        # 绘制星位
        for x in [3, 7, 11]:
            for y in [3, 7, 11]:
                self.canvas.create_oval(
                    self.margin + x * self.cell_size - 4,
                    self.margin + y * self.cell_size - 4,
                    self.margin + x * self.cell_size + 4,
                    self.margin + y * self.cell_size + 4,
                    fill='black'
                )
    
    def handle_click(self, event):
        # 只在人类玩家回合处理点击
        if self.game.current_player != 1 or self.game.game_over:
            return
        
        x = round((event.x - self.margin) / self.cell_size)
        y = round((event.y - self.margin) / self.cell_size)
        
        if 0 <= x < self.size and 0 <= y < self.size and self.game.board[y][x] == 0:
            self.make_human_move(x, y)
            
            # 如果不是游戏结束且轮到AI，则让AI下棋
            if not self.game.game_over and self.game.current_player == self.ai_player.player:
                self.root.after(500, self.make_ai_move)
    
    def make_human_move(self, x, y):
        if self.game.make_move(x, y):
            color = 'black' if self.game.board[y][x] == 1 else 'white'
            self.draw_stone(x, y, color)
            
            if self.game.game_over:
                self.show_winner()
            else:
                self.update_status()
    
    def make_ai_move(self):
        start_time = time.time()
        self.status_var.set("AI 思考中...")
        self.root.update()  # 更新 UI
        x, y = self.ai_player.get_move(self.game)
        end_time = time.time()
        print(f"AI思考时间: {end_time - start_time:.2f}秒")
        self.status_var.set(f"AI 思考时间: {end_time - start_time:.2f}秒")
        
        if x is not None and y is not None:
            self.make_human_move(x, y)  # 复用人类落子逻辑
    
    def draw_stone(self, x, y, color):
        self.canvas.create_oval(
            self.margin + x * self.cell_size - self.radius,
            self.margin + y * self.cell_size - self.radius,
            self.margin + x * self.cell_size + self.radius,
            self.margin + y * self.cell_size + self.radius,
            fill=color, outline='black', width=1
        )
    
    def show_winner(self):
        winner = "黑方" if self.game.winner == 1 else "白方"
        messagebox.showinfo("游戏结束", f"{winner}获胜！")
        self.reset_game()
    
    def update_status(self):
        self.status_var.set("黑方回合" if self.game.current_player == 1 else "白方回合")
    
    def reset_game(self):
        self.game.reset()
        self.canvas.delete("all")
        self.draw_board()
        self.update_status()
