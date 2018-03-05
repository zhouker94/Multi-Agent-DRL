# -*- coding: utf-8 -*-
# @Time    : 2018/3/5 下午2:05
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : functions.py
# @Software: PyCharm Community Edition

import constants as const


def growth_function(n):
    rg = const.REPLENISHMENT_RATE
    return rg * n * (1 - n / const.RESOURCE_CAPACITY_N_MAX)


def harvest_function(effort, n):
    return const.BETA * (effort ** const.ALPHA) * (n ** (1 - const.ALPHA))
