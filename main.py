from Coach import Coach
from ccp.CcpGame import CcpGame as Game
# from othello.pytorch.NNet import NNetWrapper as nn
from ccp.keras.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 10,
    'numEps': 10,
    'tempThreshold': 15,  # 模拟多少步之后，改取次数最多，而不是胜率最高的为best action
    'updateThreshold': 0.36, # 0.6*0.6
    'maxlenOfQueue': 200000,
    'numMCTSSims': 30,  # 往下探索多少步之后判断最佳策略,推荐25，54就肯定全部打完一局了
    'pitThreshold': 10,
    'arenaCompare': 30,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    # 'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'load_folder_file': ('./temp/', 'checkpoint_3.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__=="__main__":
    # g = Game()
    # nnet = nn(g)
    #
    # if args.load_model:
    #     nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    #
    # c = Coach(g, nnet, args)
    # if args.load_model:
    #     print("Load trainExamples from file")
    #     c.loadTrainExamples()
    # c.learn()
    # c.learnMulti()
    c = Coach(args)
    c.run()
    # _kls = Klass()
    # _kls.run()
