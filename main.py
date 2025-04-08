from game import GomokuGame
from ui import GomokuGUI
from ai import AIPlayer

def main():
    # 初始化游戏
    game = GomokuGame(size=15)
    
    # 初始化AI玩家（白棋，搜索深度3）
    ai_player = AIPlayer(player=2, depth=3)
    
    # 初始化GUI
    gui = GomokuGUI(game, ai_player)
    
    # 启动游戏
    gui.run()

if __name__ == "__main__":
    main()
