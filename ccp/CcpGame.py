from Game import Game
import copy
import numpy as np
from .CcpLogic import Poker,Player,PlayRecords,get_state,get_actions
from ccp.actions import  action_dict

class CcpGame(Game):
    def __init__(self):
        # self.g = Poker()
        pass

    def getInitBoard(self):
        """
        返回Board，这里相当于是一局牌的当前(初始)状态,对于扑克而言，
        用records来表示状态，但保存和传给神经网络训练的，要从records中抽出state
        :return:
        """
        g = Poker()

        # 发牌
        g.game_start(g.players, g.playrecords, g.cards)
        # state = get_state(self.g.playrecords, self.g.players[0])
        return g.playrecords

    def getBoardSize(self):
        # 这里返回值用于确定神经网络输入域，应该是state的大小
        return (6, 15)

    def getActionSize(self):
        # 这里返回值用于确定神经网络的Pi输出域，为所有可能的出牌组合 + 429 buyao + 430 yaobuqi
        return len(action_dict)+2

    def getNextState(self, board, player, action):
        g = Poker()
        b = copy.deepcopy(board)
        g.initFromRecords(b)
        g.execute_move(action, player)

        return (g.playrecords, g.i)

    def getValidMoves(self, board, player):
        g = Poker()
        b = copy.deepcopy(board)
        g.initFromRecords(b)
        next_move_types, next_moves = g.get_next_moves()
        actions = get_actions(next_moves, g.last_move != "start")
        # action to one-hot
        actions_one_hot = np.zeros(self.getActionSize())
        for k in range(len(actions)):
            actions_one_hot[actions[k]] = 1

        return actions_one_hot

    def getGameEnded(self, board, player):
        g = Poker()
        b = copy.deepcopy(board)
        g.initFromRecords(b)
        return g.winner

    def getCanonicalForm(self, board, player):
        g = Poker()
        b = copy.deepcopy(board)
        g.initFromRecords(b)
        # state = get_state(board, board.player)
        return g.playrecords

    def getSymmetries(self, board, pi):
        g = Poker()
        b = copy.deepcopy(board)
        g.initFromRecords(b)
        state = get_state(b, b.player)
        l = [(state, pi)]
        return l

    def stringRepresentation(self, board):
        # 为某个状态生成一个对应字符key
        # 这里的状态应该只包括当前玩家可见状态
        state = board.save_to_state()
        # state = get_state(board, board.player)
        return state.tostring()

def display(board):
    print(board.save_to_state())
