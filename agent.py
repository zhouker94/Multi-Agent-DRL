#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/18 0:15
# @Author  : Hanwei Zhu
# @File    : agent.py


import os
from model import ddpg, dqn
import numpy as np


class Agent(object):
    def __init__(self, aid, config):
        self._aid = aid
        self._config = config
        self._obs = np.zeros(
            (self._config["time_steps"], self._config["state_space"])
        )
        self._step_counter = 0

        if config["model_name"] == "DQN":
            self._model = dqn.DQNModel(aid, config)
        elif config["model_name"] == "DDPG":
            self._model = ddpg.DDPGModel(aid, config)

    def learn(self):
        self._model.fit()

    def remember(self, state, action, reward, next_state):
        self._model.save_transition(state, action, reward, next_state)

    def act(self, state, epsilon=0, **kwargs):
        self._obs[self._step_counter % self._config["time_steps"], :] = \
            np.asarray(state)
        self._step_counter += 1

        action = self._model.predict(
            np.expand_dims(
                np.mean(self._obs, axis=0),
                axis=0
            ),
            epsilon,
            **kwargs
        )

        if action < self._config["action_lower_bound"]:
            action = self._config["action_lower_bound"]

        return action

    def save(self, save_model_path):
        """
        Save Tensorflow model
        """
        model_path = os.path.join(
            save_model_path,
            self._aid
        )

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        save_path = self._model.save_model(model_path)
        print("Model saved in path: %s" % save_path)
        self._model.close()
