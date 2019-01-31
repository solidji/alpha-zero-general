from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle

from multiprocessing import Pool, Process, Queue, Manager
from ccp.keras.NNet import NNetWrapper as nn
from ccp.CcpGame import CcpGame as Game
from utils import *
import copyreg
import types

args = dotdict({
    'numIters': 10,
    'numEps': 10,
    'tempThreshold': 15,  # 模拟多少步之后，改取次数最多，而不是胜率最高的为best action
    'updateThreshold': 0.36, # 0.6*0.6
    'maxlenOfQueue': 200000,
    'numMCTSSims': 15,  # 往下探索多少步之后判断最佳策略,推荐25，54就肯定全部打完一局了
    'arenaCompare': 6,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    # 'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'load_folder_file': ('./temp/', 'checkpoint_6.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

# 封装一层函数代理，解决不能被pickle的问题
def proxy(cls_instance, i):
    return cls_instance.func(i)

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copyreg.pickle(types.MethodType, _pickle_method)

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self):
        print("Constructor ... %s" % Process().name)
        # self.game = Game()
        # self.nnet = nn(self.game)
        # self.pnet = self.nnet.__class__(self.game)  # the competitor network
        # self.args = args
        # self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    """
    这部分是用于测试多进程的
    """
    # def __init__(self):
    #     print("Constructor Run ... %s" % Process().name)
    def __del__(self):
        print("... Destructor %s" % Process().name)

    def func(self, x):
        game = Game()
        nnet = nn(game)
        pnet = nnet.__class__(game)
        mcts = MCTS(game, nnet, args)
        print(x * x)

        trainExamples = []
        board = game.getInitBoard()
        board.show("a new game start %s" % x)
        curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = game.getCanonicalForm(board, curPlayer)
            temp = int(episodeStep < args.tempThreshold)

            pi = mcts.getActionProb(canonicalBoard, temp=temp)
            sym = game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = game.getNextState(board, curPlayer, action)

            r = game.getGameEnded(board, curPlayer)

            if r != 0:
                board.show("a new game end %s" % x)
                return [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]


    def run(self):
        pool = Pool(processes=3)
        for num in range(8):
            #pool.apply_async(self.func, args=(num,))
            pool.apply_async(self.func, args=(num, ))
        pool.close()
        pool.join()
        print('All subprocesses done.')


    def learnMulti(self, cores=10):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, args.numIters + 1):
            # bookkeeping
            print('------ITER Multi' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=args.maxlenOfQueue)

                eps_time = AverageMeter()
                bar = Bar('Self Play', max=args.numEps)
                end_time = time.time()
                results = []
                p = Pool(processes=10)

                def update(result):
                    results.append(result)
                    eps_time.update(time.time() - end_time)
                    # end_time = time.time()
                    bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                        eps=eps + 1, maxeps=args.numEps, et=eps_time.avg,
                        total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()

                for eps in range(args.numEps):
                    # self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    # result = (p.apply_async(self, args=(self,)))
                    p.apply_async(self.func, args=(eps, ), callback=update)
                    # results.append(result)
                    # iterationTrainExamples += self.executeEpisode()

                    # bookkeeping + plot progress
                    # eps_time.update(time.time() - end)
                    # end = time.time()
                    # bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                    #     eps=eps + 1, maxeps=args.numEps, et=eps_time.avg,
                    #     total=bar.elapsed_td, eta=bar.eta_td)
                    # bar.next()

                print('Waiting for all subprocesses done...')
                p.close()
                p.join()
                bar.finish()
                print('All subprocesses done.')
                for res in results:
                    # print(":::", res.get())
                    iterationTrainExamples += res.get()

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
                      " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.game = Game()
            self.args = args
            self.nnet = nn(self.game)
            self.pnet = self.nnet.__class__(self.game)
            self.mcts = MCTS(self.game, self.nnet, self.args)

            self.saveTrainExamples(i - 1)

            # shuffle examlpes before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
            p1mcts = MCTS(self.game, self.pnet, args)
            p2mcts = MCTS(self.game, self.pnet, args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, args)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(p1mcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(p2mcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            p1wins, p2wins, nwins, draws = arena.playGames(args.arenaCompare)

            print('NEW/PREV WINS : %d / %d / %d ; DRAWS : %d' % (nwins, p1wins, p2wins, draws))
            if p1wins + p2wins + nwins > 0 and float(nwins) / (p1wins + p2wins + nwins) < args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
    """
    这部分是用于测试多进程的
    """

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """

        trainExamples = []
        board = self.game.getInitBoard()
        board.show("a new game start")
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                board.show("a new game end")
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        self.game = Game()
        self.args = args
        self.nnet = nn(self.game)
        self.pnet = self.nnet.__class__(self.game)
        self.mcts = MCTS(self.game, self.nnet, self.args)

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                        eps=eps + 1, maxeps=self.args.numEps, et=eps_time.avg,
                        total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
                      " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examlpes before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            p1wins, p2wins, nwins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d / %d ; DRAWS : %d' % (nwins, p1wins, p2wins, draws))
            if p1wins + p2wins + nwins > 0 and float(nwins) / (p1wins + p2wins + nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')



    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

if __name__=="__main__":
    c = Coach()
    # c.run()
    # c.learnMulti()
    c.learn()