# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : test_rps.py
#
#* Purpose :
#
#* Creation Date : 10-06-2024
#
#* Last Modified : Sat 15 Jun 2024 04:55:46 PM IST
#
#* Created By : Yaay Nands
#_._._._._._._._._._._._._._._._._._._._._.#
import tragedy_commons as tc
import axelrod as axl
from pettingzoo.test import test_save_obs

env = tc.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
test_save_obs(env)
env.close()

