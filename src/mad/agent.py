#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/18 0:15
# @Author  : Hanwei Zhu
# @File    : agent.py


import os
import uuid
import numpy as np

from .model import ddpg, dqn


class Agent:
    def __init__(self, config, ckpt_path=None):
        self._uid = str(uuid.uuid4())[:8]
        self._config = config
        self._obs = np.zeros(
            (self._config["time_steps"], self._config["state_space"])
        )
        self._step_counter = 0

        if self._config["model_name"] == "DQN":
            self._model = dqn.DQNModel(
                self._uid, self._config, ckpt_path)
        elif self._config["model_name"] == "DDPG":
            self._model = ddpg.DDPGModel(
                self._uid, self._config, ckpt_path)

    def learn(self):
        self._model.fit()

    def remember(self, state, action, reward, next_state):
        self._model.save_transition(state, action, reward, next_state)

    def act(self, state, epsilon=0, **kwargs):
        self._obs[self._step_counter % self._config["time_steps"], :] = state
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

    def close(self, save_model_path=None):
        """
        Save Tensorflow model
        """
        if save_model_path:
            model_path = os.path.join(
                save_model_path,
                self._uid
            )

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            save_path = self._model.save_model(model_path)
            print("Model saved in path: {}".format(save_path))

        self._model.close()
