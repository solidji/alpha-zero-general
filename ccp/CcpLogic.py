# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import print_function
from __future__ import absolute_import
# from .gameutil import card_show, choose, game_init
import numpy as np
from ccp.actions import action_dict
# from rl.init_model import model_init

############################################
#                 游戏类                   #
############################################
class Poker(object):

    def __init__(self, my_config=None):
        # 初始化一副扑克牌类
        self.cards = Cards()

        # play相关参数
        self.end = False
        self.last_move_type = self.last_move = "start"
        self.playround = 1
        self.i = 0
        self.yaobuqis = []

        # choose模型
        self.models = ["random", "random", "random"]

        self.my_config = my_config
        self.actions_lookuptable = action_dict

        # 初始化players
        self.players = []
        self.players.append(Player(1, game=self))
        self.players.append(Player(2, game=self))
        self.players.append(Player(3, game=self))
        # self.players.append(Player(3, self.models[2], self.my_config, self, RL))

        # 初始化扑克牌记录类
        self.playrecords = PlayRecords()


    # 发牌
    def game_start(self, players, playrecords, cards, train=True):

        if train:
            # 洗牌
            np.random.shuffle(cards.cards)
            # 排序
            p1_cards = cards.cards[:18]
            p1_cards.sort(key=lambda x: x.rank)
            p2_cards = cards.cards[18:36]
            p2_cards.sort(key=lambda x: x.rank)
            p3_cards = cards.cards[36:]
            p3_cards.sort(key=lambda x: x.rank)
            players[0].cards_left = playrecords.cards_left1 = p1_cards
            players[1].cards_left = playrecords.cards_left2 = p2_cards
            players[2].cards_left = playrecords.cards_left3 = p3_cards
        else:
            # 洗牌
            np.random.shuffle(cards.cards)
            # 排序
            p1_cards = cards.cards[:20]
            p1_cards.sort(key=lambda x: x.rank)
            p2_cards = cards.cards[20:37]
            p2_cards.sort(key=lambda x: x.rank)
            p3_cards = cards.cards[37:]
            p3_cards.sort(key=lambda x: x.rank)
            players[0].cards_left = playrecords.cards_left1 = p1_cards
            players[1].cards_left = playrecords.cards_left2 = p2_cards
            players[2].cards_left = playrecords.cards_left3 = p3_cards


    # 返回扑克牌记录类
    def get_record(self):
        web_show = WebShow(self.playrecords)
        # return jsonpickle.encode(web_show, unpicklable=False)
        return web_show

    # 返回当前出牌玩家i可选的下次出牌列表
    def get_next_moves(self):
        next_move_types, next_moves = self.players[self.i].get_moves(self.last_move_type, self.last_move,
                                                                     self.playrecords)
        return next_move_types, next_moves


    def get_next_move(self, action):
        """
        进行一局随机游戏，返回胜者ID
        :param action:
        :return:
        """
        while (self.i <= 2):
            # if self.i != 0:
            #     self.get_next_moves()
            self.last_move_type, self.last_move, self.end, self.yaobuqi = self.players[self.i].play(self.last_move_type,
                                                                                                    self.last_move,
                                                                                                    self.playrecords,
                                                                                                    action)
            if self.yaobuqi:
                self.yaobuqis.append(self.i)
            else:
                self.yaobuqis = []
            # 都要不起
            if len(self.yaobuqis) == 2:
                self.yaobuqis = []
                self.last_move_type = self.last_move = "start"
            if self.end:
                self.playrecords.winner = self.i + 1
                break
            self.i = (self.i + 1) % 3
        # 一轮结束
        self.playround = self.playround + 1
        self.i = 0
        return self.playrecords.winner, self.end

    def execute_move(self, action, player):
        '''
        在当前牌局情况下，选择action执行一步
        :param action:
        :param player:
        :return:
        '''
        # 需要补检查action是否valid，player是否0-2

        # if action in [429, 430]:
        #     action_id = action
        # else:
        #     action_id = self.actions.index(action)
        #

        #     # 训练model
        #     elif model == "rl":
        #     if action[3][action[2]] == 429:
        #         return "buyao", []
        #     elif action[3][action[2]] == 430:
        #         return "yaobuqi", []
        #     else:
        #         return action[0][action[2]], action[1][action[2]]

        # self.i = self.playrecords.player
        self.last_move_type, self.last_move, self.end, self.yaobuqi = \
            self.players[self.i].play(self.last_move_type, self.last_move, self.playrecords, action)
        if self.yaobuqi:
            self.yaobuqis.append(self.i)
        else:
            self.yaobuqis = []
        # 都要不起
        if len(self.yaobuqis) == 2:
            self.yaobuqis = []
            self.last_move_type = self.last_move = "start"
        if self.end:
            self.playrecords.winner = self.i + 1
        self.i = (self.i + 1) % 3

        return self.playrecords.winner, self.end

    def initFromRecords(self, playrecords):
        '''
        通过牌局记录恢复一个Poker类到当时的状态
        '''

        # 初始化扑克牌记录类
        self.playrecords = playrecords
        # 胜利者/当前出牌玩家,ps: self.i从0开始，player从1开始
        self.winner = playrecords.winner
        self.i = playrecords.player % 3

        # 3个玩家剩余手牌
        self.players[0].cards_left = playrecords.cards_left1
        self.players[1].cards_left = playrecords.cards_left2
        self.players[2].cards_left = playrecords.cards_left3

        # 上一手牌
        if len(playrecords.records) == 0: # 刚开局
            self.last_move_type = self.last_move = "start"
        elif len(playrecords.records) == 1: # (地主)打了一手牌
            self.last_move = playrecords.records[0][2]
            self.last_move_type = playrecords.records[0][1]
        elif len(playrecords.records) == 2: # 打了两手牌
            if playrecords.records[1][1] == "yaobuqi" or playrecords.records[1][1] == "buyao":
                self.last_move_type = playrecords.records[0][1]
                self.last_move = playrecords.records[0][2]
            else:
                self.last_move_type = playrecords.records[1][1]
                self.last_move = playrecords.records[1][2]
        elif len(playrecords.records) > 2: # 至少打了一轮牌
            if playrecords.records[-1][1] == "yaobuqi" or playrecords.records[-1][1] == "buyao": # 上家不要
                if playrecords.records[-2][1] == "yaobuqi" or playrecords.records[-2][1] == "buyao": # 上上家也不要
                    self.last_move_type = self.last_move = "start"
                else:
                    self.last_move_type = playrecords.records[-2][1] # 上家不要但上上家要了
                    self.last_move = playrecords.records[-2][2] # 上家不要但上上家要了
            else:
                self.last_move_type = playrecords.records[-1][1] # 上家要了
                self.last_move = playrecords.records[-1][2]

        return self
############################################
#              扑克牌相关类                 #
############################################
class Cards(object):
    """
    一副扑克牌类,54张排,abcd四种花色,小王14-a,大王15-a
    """

    def __init__(self):
        # 初始化扑克牌类型
        self.cards_type = ['1-a-12', '1-b-12', '1-c-12', '1-d-12',
                           '2-a-13', '2-b-13', '2-c-13', '2-d-13',
                           '3-a-1', '3-b-1', '3-c-1', '3-d-1',
                           '4-a-2', '4-b-2', '4-c-2', '4-d-2',
                           '5-a-3', '5-b-3', '5-c-3', '5-d-3',
                           '6-a-4', '6-b-4', '6-c-4', '6-d-4',
                           '7-a-5', '7-b-5', '7-c-5', '7-d-5',
                           '8-a-6', '8-b-6', '8-c-6', '8-d-6',
                           '9-a-7', '9-b-7', '9-c-7', '9-d-7',
                           '10-a-8', '10-b-8', '10-c-8', '10-d-8',
                           '11-a-9', '11-b-9', '11-c-9', '11-d-9',
                           '12-a-10', '12-b-10', '12-c-10', '12-d-10',
                           '13-a-11', '13-b-11', '13-c-11', '13-d-11',
                           '14-a-14', '15-a-15']
        # 初始化扑克牌类
        self.cards = self.get_cards()

    # 初始化扑克牌类
    def get_cards(self):
        cards = []
        for card_type in self.cards_type:
            cards.append(Card(card_type))
        # 打乱顺序
        # np.random.shuffle(cards)
        return cards


class Card(object):
    """
    扑克牌类
    """

    def __init__(self, card_type):
        self.card_type = card_type
        # 名称
        self.name = self.card_type.split('-')[0]
        # 花色
        self.color = self.card_type.split('-')[1]
        # 大小
        self.rank = int(self.card_type.split('-')[2])

    # 判断大小
    def bigger_than(self, card_instance):
        if (self.rank > card_instance.rank):
            return True
        else:
            return False


class PlayRecords(object):
    """
    扑克牌记录类
    """

    def __init__(self):
        # 每个玩家当前手牌
        self.cards_left1 = []
        self.cards_left2 = []
        self.cards_left3 = []

        # 每个玩家每一手可能出牌组合的记录,这里不会有'buyao'和'yaobuqi'
        self.next_moves1 = []
        self.next_moves2 = []
        self.next_moves3 = []

        # 每个玩家每一手出牌记录，包括'buyao'和'yaobuqi'
        self.next_move1 = []
        self.next_move2 = []
        self.next_move3 = []

        # 3个玩家整局牌的完整出牌记录，[player, move],其中move包括'buyao'和'yaobuqi'
        self.records = []

        # 胜利者
        # winner=0,1,2,3 0表示未结束,1,2,3表示winner
        self.winner = 0

        # records里最后出牌的玩家id
        self.player = 0

    # 展示
    def show(self, info):
        print(info)
        card_show(self.cards_left1, "player 1", 1)
        card_show(self.cards_left2, "player 2", 1)
        card_show(self.cards_left3, "player 3", 1)
        # card_show(self.records, "record", 3)

    # 保存当前状态为np.array形态的当前玩家视角state
    def save_to_state(self):
        i = self.player % 3 + 1
        return get_state(self, i)

    # 从state中部分还原
    def load_from_state(self):
        pass


############################################
#              出牌相关类                   #
############################################
class Moves(object):
    """
    出牌类,单,对,三,三带一,三带二,顺子,炸弹
    """

    def __init__(self):
        # 出牌信息
        self.dan = []
        self.dui = []
        self.san = []
        self.san_dai_yi = []
        self.san_dai_er = []
        self.bomb = []
        self.shunzi = []

        # 牌数量信息
        self.card_num_info = {}
        # 牌顺序信息,计算顺子
        self.card_order_info = []
        # 王牌信息
        self.king = []

        # 下次出牌
        self.next_moves = []
        # 下次出牌类型
        self.next_moves_type = []

    # 获取全部出牌列表,赋值牌型信息
    def get_total_moves(self, cards_left):

        # 统计牌数量/顺序/王牌信息
        for i in cards_left:
            # 王牌信息
            if i.rank in [14, 15]:
                self.king.append(i)
            # 数量
            tmp = self.card_num_info.get(i.rank, [])
            if len(tmp) == 0:
                self.card_num_info[i.rank] = [i]
            else:
                self.card_num_info[i.rank].append(i)
            # 顺序
            if i.rank in [13, 14, 15]:  # 不统计2,小王,大王
                continue
            elif len(self.card_order_info) == 0:
                self.card_order_info.append(i)
            elif i.rank != self.card_order_info[-1].rank:
                self.card_order_info.append(i)

        # 王炸
        if len(self.king) == 2:
            self.bomb.append(self.king)

        # 出单,出对,出三,炸弹(考虑拆开)
        for k, v in self.card_num_info.items():
            if len(v) == 1:
                self.dan.append(v)
        for k, v in self.card_num_info.items():
            if len(v) == 2:
                self.dui.append(v)
                self.dan.append(v[:1])
        for k, v in self.card_num_info.items():
            if len(v) == 3:
                self.san.append(v)
                self.dui.append(v[:2])
                self.dan.append(v[:1])
        for k, v in self.card_num_info.items():
            if len(v) == 4:
                self.bomb.append(v)
                self.san.append(v[:3])
                self.dui.append(v[:2])
                self.dan.append(v[:1])

        # 三带一,三带二
        for san in self.san:
            # if self.dan[0][0].name != san[0].name:
            #    self.san_dai_yi.append(san+self.dan[0])
            # if self.dui[0][0].name != san[0].name:
            #    self.san_dai_er.append(san+self.dui[0])
            for dan in self.dan:
                # 防止重复
                if dan[0].name != san[0].name:
                    self.san_dai_yi.append(san + dan)
            for dui in self.dui:
                # 防止重复
                if dui[0].name != san[0].name:
                    self.san_dai_er.append(san + dui)

                    # 获取最长顺子
        max_len = []
        for i in self.card_order_info:
            if i == self.card_order_info[0]:
                max_len.append(i)
            elif max_len[-1].rank == i.rank - 1:
                max_len.append(i)
            else:
                if len(max_len) >= 5:
                    self.shunzi.append(max_len)
                max_len = [i]
        # 最后一轮
        if len(max_len) >= 5:
            self.shunzi.append(max_len)
            # 拆顺子
        shunzi_sub = []
        for i in self.shunzi:
            len_total = len(i)
            n = len_total - 5
            # 遍历所有可能顺子长度
            while (n > 0):
                len_sub = len_total - n
                j = 0
                while (len_sub + j <= len(i)):
                    # 遍历该长度所有组合
                    shunzi_sub.append(i[j:len_sub + j])
                    j = j + 1
                n = n - 1
        self.shunzi.extend(shunzi_sub)

    # 获取下次出牌列表
    def get_next_moves(self, last_move_type, last_move):
        # 没有last,全加上,除了bomb最后加
        if last_move_type == "start":
            moves_types = ["dan", "dui", "san", "san_dai_yi", "san_dai_er", "shunzi"]
            i = 0
            for move_type in [self.dan, self.dui, self.san, self.san_dai_yi,
                              self.san_dai_er, self.shunzi]:
                for move in move_type:
                    self.next_moves.append(move)
                    self.next_moves_type.append(moves_types[i])
                i = i + 1
        # 出单
        elif last_move_type == "dan":
            for move in self.dan:
                # 比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)
                    self.next_moves_type.append("dan")
        # 出对
        elif last_move_type == "dui":
            for move in self.dui:
                # 比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)
                    self.next_moves_type.append("dui")
        # 出三个
        elif last_move_type == "san":
            for move in self.san:
                # 比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)
                    self.next_moves_type.append("san")
        # 出三带一
        elif last_move_type == "san_dai_yi":
            for move in self.san_dai_yi:
                # 比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)
                    self.next_moves_type.append("san_dai_yi")
        # 出三带二
        elif last_move_type == "san_dai_er":
            for move in self.san_dai_er:
                # 比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)
                    self.next_moves_type.append("san_dai_er")
        # 出炸弹
        elif last_move_type == "bomb":
            for move in self.bomb:
                # 比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)
                    self.next_moves_type.append("bomb")
        # 出顺子
        elif last_move_type == "shunzi":
            for move in self.shunzi:
                # 相同长度
                if len(move) == len(last_move):
                    # 比last大
                    if move[0].bigger_than(last_move[0]):
                        self.next_moves.append(move)
                        self.next_moves_type.append("shunzi")
        else:
            print("last_move_type_wrong")

        # 除了bomb,都可以出炸
        if last_move_type != "bomb":
            for move in self.bomb:
                self.next_moves.append(move)
                self.next_moves_type.append("bomb")

        return self.next_moves_type, self.next_moves

    # 展示
    def show(self, info):
        print(info)
        # card_show(self.dan, "dan", 2)
        # card_show(self.dui, "dui", 2)
        # card_show(self.san, "san", 2)
        # card_show(self.san_dai_yi, "san_dai_yi", 2)
        # card_show(self.san_dai_er, "san_dai_er", 2)
        # card_show(self.bomb, "bomb", 2)
        # card_show(self.shunzi, "shunzi", 2)
        # card_show(self.next_moves, "next_moves", 2)


############################################
#              玩家相关类                   #
############################################
class Player(object):
    """
    player类
    """

    def __init__(self, player_id, model="random", my_config=None, game=None, RL=None):
        self.player_id = player_id
        self.cards_left = []
        # 出牌模式
        self.model = model
        # RL_model
        self.game = game
        self.my_config = my_config

        self.RL = RL

    # 展示
    def show(self, info):
        self.total_moves.show(info)
        card_show(self.next_move, "next_move", 1)
        # card_show(self.cards_left, "card_left", 1)

    # 根据next_move同步cards_left
    def record_move(self, playrecords):
        # 记录出牌者
        playrecords.player = self.player_id
        # playrecords中records记录[id,next_move]
        if self.next_move_type in ["yaobuqi", "buyao"]:
            self.next_move = self.next_move_type
            playrecords.records.append([self.player_id, self.next_move_type, self.next_move])
        else:
            playrecords.records.append([self.player_id, self.next_move_type, self.next_move])
            for i in self.next_move:
                self.cards_left.remove(i)
        # 同步playrecords
        if self.player_id == 1:
            playrecords.cards_left1 = self.cards_left
            playrecords.next_moves1.append(self.next_moves) #yaobuqi的时候next_moves与tpyes为空list[]，buyao的时候不为空但不包括'buyao'
            playrecords.next_move1.append(self.next_move) #而next_move与type都为'yaobuqi'或'buyao'
        elif self.player_id == 2:
            playrecords.cards_left2 = self.cards_left
            playrecords.next_moves2.append(self.next_moves)
            playrecords.next_move2.append(self.next_move)
        elif self.player_id == 3:
            playrecords.cards_left3 = self.cards_left
            playrecords.next_moves3.append(self.next_moves)
            playrecords.next_move3.append(self.next_move)
        # 是否牌局结束
        end = False
        if len(self.cards_left) == 0:
            playrecords.winner = self.player_id
            end = True
        return end

    # 选牌
    def get_moves(self, last_move_type, last_move, playrecords):
        # 所有出牌可选列表
        self.total_moves = Moves()
        # 获取全部出牌列表
        self.total_moves.get_total_moves(self.cards_left)
        # 获取下次出牌列表
        self.next_move_types, self.next_moves = self.total_moves.get_next_moves(last_move_type, last_move)
        # 返回下次出牌列表
        return self.next_move_types, self.next_moves

    # 出牌
    def play(self, last_move_type, last_move, playrecords, action=None):
        '''
        按给定action走一步，记录到records，判断是否结束，
        没有给定action则随机走一步

        :param last_move_type:
        :param last_move:
        :param playrecords:
        :param action:
        :return:
        '''
        # 主动调用一下，初始化self.next_move_types, self.next_moves
        self.get_moves(last_move_type, last_move, playrecords)
        # print("choose action: ", self.player_id, action)
        if action != None:
            if action == 429:
                self.next_move_type = self.next_move = "buyao"
            elif action == 430:
                self.next_move_type = self.next_move = "yaobuqi"
            else:
                actions = get_actions(self.next_moves, False)
                # print("in actions: ", action, actions)
                for a in actions:
                    if a == action:
                        self.next_move = self.next_moves[actions.index(a)]
                        self.next_move_type = self.next_move_types[actions.index(a)]
                        break

            # self.next_move_type, self.next_move = get_move(action)

        else:
            # 在next_moves中选出一种出牌
            self.next_move_type, self.next_move = choose_random(next_move_types=self.next_move_types,
                                                            next_moves=self.next_moves,
                                                            last_move_type=last_move_type)
        # 记录
        end = self.record_move(playrecords)
        # 展示
        # self.show("Player " + str(self.player_id))
        # 要不起&不要
        yaobuqi = False
        if self.next_move_type in ["yaobuqi", "buyao"]:
            yaobuqi = True # 连续2个不要或要不起，重置last_move为"start"
            self.next_move_type = last_move_type # 要不起或不要的时候，将上一步动作保存并原样返回
            self.next_move = last_move

        return self.next_move_type, self.next_move, end, yaobuqi


############################################
#               网页展示类                 #
############################################
class WebShow(object):
    """
    网页展示类
    """

    def __init__(self, playrecords):

        # 胜利者
        self.winner = playrecords.winner

        # 剩余手牌
        self.cards_left1 = []
        for i in playrecords.cards_left1:
            self.cards_left1.append(i.name + i.color)
        self.cards_left2 = []
        for i in playrecords.cards_left2:
            self.cards_left2.append(i.name + i.color)
        self.cards_left3 = []
        for i in playrecords.cards_left3:
            self.cards_left3.append(i.name + i.color)

            # 可能出牌
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


############################################
#                   LR相关                 #
############################################
def get_state(playrecords, player):
    '''
    定义当前state为6x13的矩阵，分别是：
    自己手牌+自己打过的牌+下家打过的牌+上家打过的牌+下家上一手牌+上家上一手牌
    :param playrecords:
    :param player:
    :return:
    '''
    state = np.zeros((6, 15), dtype=int)

    if player == 1 or player == 0:
        for i in playrecords.cards_left1:
            state[0][i.rank - 1] += 1
        for cards in playrecords.next_move1:
            if cards in ["buyao", "yaobuqi"]:
                continue
            for card in cards:
                state[1][card.rank - 1] += 1
        for cards in playrecords.next_move2:
            if cards in ["buyao", "yaobuqi"]:
                continue
            for card in cards:
                state[2][card.rank - 1] += 1
        for cards in playrecords.next_move3:
            if cards in ["buyao", "yaobuqi"]:
                continue
            for card in cards:
                state[3][card.rank - 1] += 1
        if playrecords.next_move2:
            cards = playrecords.next_move2[-1]
            if cards not in ["buyao", "yaobuqi"]:
                for card in cards:
                    state[4][card.rank - 1] += 1
        if playrecords.next_move3:
            cards = playrecords.next_move3[-1]
            if cards not in ["buyao", "yaobuqi"]:
                for card in cards:
                    state[5][card.rank - 1] += 1

    elif player == 2:
        for i in playrecords.cards_left2:
            state[0][i.rank - 1] += 1
        for cards in playrecords.next_move2:
            if cards in ["buyao", "yaobuqi"]:
                continue
            for card in cards:
                state[1][card.rank - 1] += 1
        for cards in playrecords.next_move3:
            if cards in ["buyao", "yaobuqi"]:
                continue
            for card in cards:
                state[2][card.rank - 1] += 1
        for cards in playrecords.next_move1:
            if cards in ["buyao", "yaobuqi"]:
                continue
            for card in cards:
                state[3][card.rank - 1] += 1
        if playrecords.next_move3:
            cards = playrecords.next_move3[-1]
            if cards not in ["buyao", "yaobuqi"]:
                for card in cards:
                    state[4][card.rank - 1] += 1
        if playrecords.next_move1:
            cards = playrecords.next_move1[-1]
            if cards not in ["buyao", "yaobuqi"]:
                for card in cards:
                    state[5][card.rank - 1] += 1

    elif player == 3:
        for i in playrecords.cards_left3:
            state[0][i.rank - 1] += 1
        for cards in playrecords.next_move3:
            if cards in ["buyao", "yaobuqi"]:
                continue
            for card in cards:
                state[1][card.rank - 1] += 1
        for cards in playrecords.next_move1:
            if cards in ["buyao", "yaobuqi"]:
                continue
            for card in cards:
                state[2][card.rank - 1] += 1
        for cards in playrecords.next_move2:
            if cards in ["buyao", "yaobuqi"]:
                continue
            for card in cards:
                state[3][card.rank - 1] += 1
        if playrecords.next_move1:
            cards = playrecords.next_move1[-1]
            if cards not in ["buyao", "yaobuqi"]:
                for card in cards:
                    state[4][card.rank - 1] += 1
        if playrecords.next_move2:
            cards = playrecords.next_move2[-1]
            if cards not in ["buyao", "yaobuqi"]:
                for card in cards:
                    state[5][card.rank - 1] += 1
    #
    # # 手牌
    # if player == 1:
    #     cards_left = playrecords.cards_left1
    #     state[30] = len(playrecords.cards_left1)
    #     state[31] = len(playrecords.cards_left2)
    #     state[32] = len(playrecords.cards_left3)
    # elif player == 2:
    #     cards_left = playrecords.cards_left2
    #     state[30] = len(playrecords.cards_left2)
    #     state[31] = len(playrecords.cards_left3)
    #     state[32] = len(playrecords.cards_left1)
    # else:
    #     cards_left = playrecords.cards_left3
    #     state[30] = len(playrecords.cards_left3)
    #     state[31] = len(playrecords.cards_left1)
    #     state[32] = len(playrecords.cards_left2)
    # for i in cards_left:
    #     state[i.rank - 1] += 1
    # # 底牌
    # for cards in playrecords.records:
    #     if cards[1] in ["buyao", "yaobuqi"]:
    #         continue
    #     for card in cards[1]:
    #         state[card.rank - 1 + 15] += 1

    return state


def get_actions(next_moves, bStart):
    """
    0-14: 单出， 1-13，小王，大王
    15-27: 对，1-13
    28-40: 三，1-13
    41-196: 三带1，先遍历111.2，111.3，一直到131313.12
    197-352: 三带2，先遍历111.22,111.33,一直到131313.1212
    353-366: 炸弹，1111-13131313，加上王炸
    367-402: 先考虑5个的顺子，按照顺子开头从小到大进行编码，共计8+7+..+1=36
    430: yaobuqi
    429: buyao
    """
    actions_lookuptable = action_dict
    actions = []
    for cards in next_moves:
        key = []
        for card in cards:
            key.append(int(card.name))
        key.sort()
        actions.append(actions_lookuptable[str(key)])

    # yaobuqi
    if len(actions) == 0:
        actions.append(430)
    # buyao
    elif bStart:
        actions.append(429)

    return actions

def get_move(action):
    actions_lookuptable = action_dict

    if action == 429:
        next_move_type = next_move = "buyao"
    elif action == 430:
        next_move_type = next_move = "yaobuqi"
    else:
        for k, v in actions_lookuptable.items():
            if(v == action):
                print(k)
        next_move = action
        next_move_type = action

    return next_move_type, next_move

# 结合state和可以出的actions作为新的state
def combine(s, a):
    for i in a:
        s[33 + i] = 1
    return s


############################################
#                 random                    #
############################################
def choose_random(next_move_types, next_moves, last_move_type):
    # 要不起
    if len(next_moves) == 0:
        return "yaobuqi", []
    else:
        # start不能不要
        if last_move_type == "start":
            r_max = len(next_moves)
        else:
            r_max = len(next_moves) + 1
        r = np.random.randint(0, r_max)
        # 添加不要
        if r == len(next_moves):
            return "buyao", []

    return next_move_types[r], next_moves[r]

# def config():
#     self.actions_lookuptable = action_dict
#     self.dim_actions = len(self.actions_lookuptable) + 2  # 429 buyao, 430 yaobuqi
#     self.dim_states = 30 + 3 + 431  # 431为dim_actions

if __name__ == "__main__":
    g = Poker()
    g.game_start(True)
    done = False
    while (not done):
        old_state = get_state(g.playrecords, g.players[0])
        print(" -----------------------")
        print(old_state)
        print(" -----------------------")
        # next_move_types, next_moves = g.get_next_moves()
        # actions = get_actions(next_moves, g)
        winner, done = g.get_next_move(action=None)
        new_state = get_state(g.playrecords, g.players[0])

    print(" -----------------------")
    print(new_state)
    print(winner)
