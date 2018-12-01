#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/18 0:15
# @Author  : Hanwei Zhu
# @File    : agent.py


import os
from model import ddpg, dqn


class Agent(object):
    def __init__(self, aid, config, learn_mode, save_model_path):
        self._aid = aid
        self._config = config
        self._learn_mode = learn_mode
        self._save_model_path = save_model_path
        
        if config["model_name"] == "DQN":
            self._model = dqn.DQNModel(aid, config)
        elif config["model_name"] == "DDPG":
            self._model = ddpg.DDPGModel(aid, config)

    def learn(self):
        self._model.fit()

    def remember(self, state, action, reward, next_state):
        self._model.save_transition(state, action, reward, next_state)

    def act(self, state, **kwargs):
        self._model.predict(state, kwargs)

    def save(self):
        """
        Save Tensorflow model
        """
        model_path = os.path.join(self._save_model_path,
                                  self._aid)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        save_path = self._model.save_model(model_path)
        print("Model saved in path: %s" % save_path)
