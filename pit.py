import Arena
from MCTS import MCTS
from ccp.CcpGame import CcpGame as Game, display
from ccp.CcpPlayers import *
from ccp.keras.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = Game()

# all players
rp = RandomPlayer(g).play
# gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play

# nnet players
n1 = NNet(g)
# n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
n1.load_checkpoint('./temp/', 'best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


n2 = NNet(g)
# n2.load_checkpoint('./pretrained_models/othello/keras/','6x6 checkpoint_145.pth.tar')
n2.load_checkpoint('./temp/', 'checkpoint_6.pth.tar')
args2 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena.Arena(n1p, n2p, hp, g, display=display)
print(arena.playGames(3, verbose=True))
