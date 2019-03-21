import random
from typing import List

from abc import ABCMeta, abstractmethod
from collections import deque

import numpy as np

from .base.utils import Action
from learning_algorithms.network import Network

from configs.base_config import RAYS_COUNT, NETWORK, NUM_DEFAULT_INPUT_NEURONS


class Agent(metaclass=ABCMeta):
    @property
    @abstractmethod
    def rays(self):
        pass

    @abstractmethod
    def choose_action(self, sensor_info):
        pass

    @abstractmethod
    def receive_feedback(self, reward):
        pass


class SimpleCarAgent(Agent):
    def __init__(self, history_data: int = 50000):
        """
        Creates Car Agent
            :param history_data: num of stored records (results of previous steps)
        """
        self.evaluate_mode = False  # # (True) if we evaluate model, otherwise - training mode (False)
        self._rays = RAYS_COUNT  # radar beams count

        self.neural_net = Network(NETWORK, output_function=lambda x: x, output_derivative=lambda x: 1)
        self.sensor_data_history = deque([], maxlen=history_data)
        self.chosen_actions_history = deque([], maxlen=history_data)
        self.reward_history = deque([], maxlen=history_data)
        self.step = 0

    @classmethod
    def from_weights(cls, layers: List[int], weights, biases):
        """
        Creates agent by neural network params.
        """
        agent = SimpleCarAgent()
        agent._rays = weights[0].shape[1] - NUM_DEFAULT_INPUT_NEURONS
        nn = Network(layers, output_function=lambda x: x, output_derivative=lambda x: 1)

        if len(weights) != len(nn.weights):
            raise AssertionError("You provided %d weight matrices instead of %d" % (len(weights), len(nn.weights)))
        for i, (w, right_w) in enumerate(zip(weights, nn.weights)):
            if w.shape != right_w.shape:
                raise AssertionError("weights[%d].shape = %s instead of %s" % (i, w.shape, right_w.shape))
        nn.weights = weights

        if len(biases) != len(nn.biases):
            raise AssertionError("You provided %d bias vectors instead of %d" % (len(weights), len(nn.weights)))
        for i, (b, right_b) in enumerate(zip(biases, nn.biases)):
            if b.shape != right_b.shape:
                raise AssertionError("biases[%d].shape = %s instead of %s" % (i, b.shape, right_b.shape))
        nn.biases = biases

        agent.neural_net = nn

        return agent

    @classmethod
    def from_string(cls, s):
        from numpy import array  # important for eval execution
        layers, weights, biases = eval(s.replace("\n", ""), locals())
        return cls.from_weights(layers, weights, biases)

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as f:
            return cls.from_string(f.read())

    def show_weights(self):
        params = self.neural_net.sizes, self.neural_net.weights, self.neural_net.biases
        np.set_printoptions(threshold=np.nan)
        return repr(params)

    def to_file(self, filename):
        c = self.show_weights()
        f = open(filename, "w")
        f.write(c)
        f.close()

    @property
    def rays(self):
        return self._rays

    def choose_action(self, sensor_info):
        # try to predict reward for all actions that are avaliable from current state
        rewards_to_controls_map = {}

        # make discrete a set of values,
        # because we can to predict just some of them

        # you can choose another step of discretization
        for steering in np.linspace(-1, 1, 3):
            for acceleration in np.linspace(-0.75, 0.75, 3):
                action = Action(steering, acceleration)
                agent_vector_representation = np.append(sensor_info, action)
                agent_vector_representation = agent_vector_representation.flatten()[:, np.newaxis]
                predicted_reward = float(self.neural_net.feedforward(agent_vector_representation))
                rewards_to_controls_map[predicted_reward] = action

        # search for action with best reward
        rewards = list(rewards_to_controls_map.keys())
        highest_reward = max(rewards)
        best_action = rewards_to_controls_map[highest_reward]

        # Sometimes we make random action
        if (not self.evaluate_mode) and (random.random() < 0.05):
            highest_reward = rewards[np.random.choice(len(rewards))]
            best_action = rewards_to_controls_map[highest_reward]

        # следующие строки помогут вам понять, что предсказывает наша сеть
        #     print("Chosen random action w/reward: {}".format(highest_reward))
        # else:
        #     print("Chosen action w/reward: {}".format(highest_reward))

        # запомним всё, что только можно: мы хотим учиться на своих ошибках
        self.sensor_data_history.append(sensor_info)
        self.chosen_actions_history.append(best_action)

        # here we do not know what reward is
        # method receive_feedback calculates real reward from predicted action
        self.reward_history.append(0.0)

        return best_action

    def receive_feedback(self, reward, train_every=50, reward_depth=7):
        """
        Receive feedback on the latest decision,
            predicted by neural network and analyze it

        :param reward: real world reward
        :param train_every: sufficient data count for training mode
        :param reward_depth: how many actions in a row affects on reward

        """
        # считаем время жизни сети; помогает отмерять интервалы обучения
        self.step += 1

        # начиная с полной полученной истинной награды,
        # размажем её по предыдущим наблюдениям
        # чем дальше каждый раз домножая её на 1/2
        # (если мы врезались в стену - разумно наказывать не только последнее
        # действие, но и предшествующие)
        i = -1
        while len(self.reward_history) > abs(i) and abs(i) < reward_depth:
            self.reward_history[i] += reward
            reward *= 0.5
            i -= 1

        # Если у нас накопилось хоть чуть-чуть данных, давайте потренируем нейросеть
        # прежде чем собирать новые данные
        # (проверьте, что вы в принципе храните достаточно данных (параметр `history_data` в `__init__`),
        # чтобы условие len(self.reward_history) >= train_every выполнялось
        if not self.evaluate_mode and (len(self.reward_history) >= train_every) and not (self.step % train_every):
            X_train = np.concatenate([self.sensor_data_history, self.chosen_actions_history], axis=1)
            y_train = self.reward_history
            train_data = [(x[:, np.newaxis], y) for x, y in zip(X_train, y_train)]
            self.neural_net.SGD(training_data=train_data, epochs=15, mini_batch_size=train_every, eta=0.05)
