from Game import Game
import copy
from .CcpLogic import Poker,Player,PlayRecords,get_state
from ccp.actions import  action_dict

class CcpGame(Game):
    def __init__(self):
        pass

    def getInitBoard(self):
        # 返回Board，这里相当于是一局牌的当前(初始)状态,对于扑克而言，
        # 用records来表示状态，但保存和传给神经网络训练的，要从records中抽出state
        g = Poker()
        #洗牌、发牌、设置地主
        g.game_start()

        # state = get_state(g.playrecords, g.players[0])
        return g.playrecords

    def getBoardSize(self):
        # 这里返回值用于确定神经网络输入域，应该是state的大小
        return (6, 15)

    def getActionSize(self):
        # 这里返回值用于确定神经网络的Pi输出域，为所有可能的出牌组合 + 429 buyao + 430 yaobuqi
        return len(action_dict)+2

    def getNextState(self, board, player, action):
        g = Poker()
        g.initFromRecords(board)
        g.execute_move(action, player)

        return (g.playrecords, g.i)





    def getValidMoves(self, board, player):

        pass

    def getGameEnded(self, board, player):
        pass

    def getCanonicalForm(self, board, player):
        pass

    def getSymmetries(self, board, pi):
        return board

    def stringRepresentation(self, board):
        # 为某个状态生成一个对应字符key
        # 这里的状态应该只包括当前玩家可见状态
        pass

