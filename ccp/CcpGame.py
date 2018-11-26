from Game import Game
from .CcpLogic import Poker,get_state,action_dict

class CcpGame(Game):
    def __init__(self):
        pass

    def getInitBoard(self):
        #返回Board，这里相当于是一局牌的当前(初始)状态
        g = Poker()
        #洗牌、发牌、设置地主
        g.game_start(train=True)

        state = get_state(g.playrecords, g.players[0])
        return state

    def getBoardSize(self):

        return (6, 15)

    def getActionSize(self):
        return len(action_dict)+1 # 所有可能组合+buyao

    def getNextState(self, board, player, action):
        g = Poker()
        g.playrecords = board
        g.players = board
        g.execute_move(action, player)

        return (g, (player+1)%3-1)

    def getSymmetries(self, board, pi):
        return board

    def stringRepresentation(self, board):
        # 为某个状态生成一个对应字符key
        # 这里的状态应该只包括当前玩家可见状态
        pass
