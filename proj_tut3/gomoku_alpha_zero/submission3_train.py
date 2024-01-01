from __future__ import print_function
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

from gomoku_gym.submission2_mcts_pure import MCTSPlayer as MCTS_Pure
from gomoku_gym.envs.game import Board, Game

from gomoku_alpha_zero.submission3_mcts_alphaZero import MCTSPlayer
from gomoku_alpha_zero.submission3_policy_value_net_pytorch import PolicyValueNet  


class TrainPipeline():
    def __init__(self, init_model=None):
        """ init function for the class"""

        # params of the board and the game
        self.board_width = 6 # board width
        self.board_height = 6 # board height
        self.n_in_row = 4 # win by n in line (vertically, horizontally, diagonally)
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5 # a number in (0, inf) controlling the relative impact of value Q, and prior probability P, on this node's score.
        self.buffer_size = 10000 # buffer size for replaying experience
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size) # buffer
        self.play_batch_size = 1 # size of rollout for each episode
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02 # target of KL loss
        self.check_freq = 50 # frequency for check evaluation and save model
        self.game_batch_num = 750 # number of training game loop
        self.best_win_ratio = 0.0 # best evaluated win ratio
        self.WinRatio_history_X = []
        self.WinRatio_history_Y = []
        self.Loss_Y = []
        self.PolicyEntropy_Y = []
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        if init_model: # load from existing file
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping

        Description:
            We can increase the training data by simply rotating or flipping the state. In such a way,
            we can get more data to contribute to increasing the performance of training neural network.

        input params:
            play_data: type:List,  [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play rollout data for training

        input param:
            n_games: number of rollout
        """
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net by training net

        Pipeline:
            1. sample data from the deque: self.data_buffer
            2. compute action probability for original policy network
            3. train neural network in a loop given sampled data
                    loop pipeline:
                        1. call self.policy_value_net.train_step(state_batch,
                                                                mcts_probs_batch,
                                                                winner_batch,
                                                                self.learn_rate*self.lr_multiplier)
                        2. compute action probability for new trained policy network
                        3. compute kl divergence between old and new action probability
                        4. if kl > self.kl_targ * 4, break the loop for Early Stopping
            4. adjust learning rate based on kl divergence
                    if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
                        self.lr_multiplier /= 1.5 # decrease learning rate
                    elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
                        self.lr_multiplier *= 1.5 # increase learning rate
            4. return final loss and entropy


        :return:
            loss:
            entropy:
        """
        # TODO: code here
        # sample data from the deque: self.data_buffer
        ExpData_databuffer = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in ExpData_databuffer]
        mcts_probs_batch = [data[1] for data in ExpData_databuffer]
        winner_batch = [data[2] for data in ExpData_databuffer]

        # compute action probability for original policy network
        probs_previous, v_previous = self.policy_value_net.policy_value(state_batch)

        # train neural network in a loop given sampled data
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)

            # compute action probability for new trained policy network
            probs_new, v_new = self.policy_value_net.policy_value(state_batch)

            # compute kl divergence between old and new action probability
            temp_deltaAction = np.log(probs_previous + 1e-10) - np.log(probs_new + 1e-10)
            kl = np.mean(np.sum(probs_previous * temp_deltaAction, axis=1))

            # if kl > self.kl_targ * 4, break the loop for Early Stopping
            if kl > self.kl_targ * 4:  
                break

        # adjust learning rate based on kl divergence
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(("loss:{},"
                "entropy:{},").format(loss,entropy,))

        # return final loss and entropy
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """ Policy Evaluation

        Description:
            Evaluate the trained policy by playing against the pure MCTS player
            Note: this is only for monitoring the progress of training

        Pipeline:
            1. create MCTSPlayer and MCTS_Pure Player
            2. Evaluation loop
                    Pipeline:
                        1. Rollout simulation for AlphaZero vs Pure MCTS
                                winner = self.game.start_play(current_mcts_player,
                                                  pure_mcts_player,
                                                  start_player=i % 2,  # start from either Player 1 or 2 evenly
                                                  is_shown=0)
                        2. Record result
            3. compute winning ratio: win_ratio
                    winning ratio =  (winning times + 0.5 * tie times) / total times
        return:
            win_ratio
        """
        # TODO: code here
        # create MCTSPlayer and MCTS_Pure Player
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)

        # Rollout simulation for AlphaZero vs Pure MCTS
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,  # start from either Player 1 or 2 evenly
                                          is_shown=0)
            # Record result
            win_cnt[winner] += 1

        # compute winning ratio: win_ratio
        win_ratio = (win_cnt[1] + 0.5*win_cnt[-1]) / n_games

        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))

        return win_ratio

    def run(self):
        """run the training pipeline

        Descriptions:
            train alpha zero in a loop.
            loop size: self.game_batch_num

        loop pipline:
            1. collect self-play data by rollouts
            2. policy update by sampled training data
            3. evaluated model performance (in a fixed frequency)
            4. save model (in a fixed frequency)
            5. evaluation result
        Plot

        """
        try:
            # TODO: code here
            # collect self-play data by rollouts
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)

                print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))

                # policy update by sampled training data
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()

                # evaluated model performance (in a fixed frequency)
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))

                    win_ratio = self.policy_evaluate()

                    # save model (in a fixed frequency)
                    self.policy_value_net.save_model('./current_policy.model')


                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        
                        self.best_win_ratio = win_ratio

                        # record win ratio for plot
                        self.WinRatio_history_X.append(i+1)
                        self.WinRatio_history_Y.append(win_ratio)
                        self.Loss_Y.append(loss)
                        self.PolicyEntropy_Y.append(entropy)

                        self.policy_value_net.save_model('./best_policy.model')

                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                        
        except KeyboardInterrupt:
            print('\n\rquit')

        # plot evaluation result
        # TODO: code here
        if i == self.game_batch_num - 1:
            plt.plot(self.WinRatio_history_X, self.WinRatio_history_Y)
            plt.xlabel('no. of self-play games')
            plt.ylabel('winning ratio')
            plt.title('Evaluation curve of the trained model against a pure MCTS model')
            plt.savefig('evaluation_curve.png')
            plt.show()
            
            plt.plot(self.WinRatio_history_X, self.Loss_Y)
            plt.xlabel('no. of self-play games')
            plt.ylabel('loss')
            plt.title('Loss of the trained model')
            plt.savefig('loss.png')
            plt.show()
            
            plt.plot(self.WinRatio_history_X, self.PolicyEntropy_Y)
            plt.xlabel('no. of self-play games')
            plt.ylabel('policy entropy')
            plt.title('policy entropy of the trained model')
            plt.savefig('policy_entropy.png')
            plt.show()
            



if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()