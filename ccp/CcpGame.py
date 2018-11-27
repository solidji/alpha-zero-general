from Game import Game
import copy
from .CcpLogic import Poker,Player,PlayRecords,get_state,action_dict

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

    def initFromRecords(self, playrecords):
        # 通过牌局记录初始化一个Poker类
        g = Poker()
        # 胜利者
        g.winner = playrecords.winner

        # 初始化players
        g.players = []
        g.players.append(Player(1, game=g))
        g.players.append(Player(2, game=g))
        g.players.append(Player(3, game=g))

        # 初始化扑克牌记录类
        g.playrecords = copy.copy(playrecords)

        # 剩余手牌
        g.players[0].cards_left = playrecords.cards_left1
        g.players[1].cards_left = playrecords.cards_left2
        g.players[2].cards_left = playrecords.cards_left3

        # 当前这一手可选的出牌组合
        self.next_moves1 = []
        if len(playrecords.next_moves1) != 0:
            next_moves = playrecords.next_moves1[-1]
            for move in next_moves:
                cards = []
                for card in move:
                    cards.append(card.name + card.color)
                self.next_moves1.append(cards)
        self.next_moves2 = []
        if len(playrecords.next_moves2) != 0:
            next_moves = playrecords.next_moves2[-1]
            for move in next_moves:
                cards = []
                for card in move:
                    cards.append(card.name + card.color)
                self.next_moves2.append(cards)
        self.next_moves3 = []
        if len(playrecords.next_moves3) != 0:
            next_moves = playrecords.next_moves3[-1]
            for move in next_moves:
                cards = []
                for card in move:
                    cards.append(card.name + card.color)
                self.next_moves3.append(cards)

                # 出牌
        self.next_move1 = []
        if len(playrecords.next_move1) != 0:
            next_move = playrecords.next_move1[-1]
            if next_move in ["yaobuqi", "buyao"]:
                self.next_move1.append(next_move)
            else:
                for card in next_move:
                    self.next_move1.append(card.name + card.color)
        self.next_move2 = []
        if len(playrecords.next_move2) != 0:
            next_move = playrecords.next_move2[-1]
            if next_move in ["yaobuqi", "buyao"]:
                self.next_move2.append(next_move)
            else:
                for card in next_move:
                    self.next_move2.append(card.name + card.color)
        self.next_move3 = []
        if len(playrecords.next_move3) != 0:
            next_move = playrecords.next_move3[-1]
            if next_move in ["yaobuqi", "buyao"]:
                self.next_move3.append(next_move)
            else:
                for card in next_move:
                    self.next_move3.append(card.name + card.color)

                    # 记录
        self.records = []
        for i in playrecords.records:
            tmp = []
            tmp.append(i[0])
            tmp_name = []
            # 处理要不起
            try:
                for j in i[1]:
                    tmp_name.append(j.name + j.color)
                tmp.append(tmp_name)
            except:
                tmp.append(i[1])
            self.records.append(tmp)

        pass