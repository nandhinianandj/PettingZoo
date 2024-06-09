# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : test_rps.py
#
#* Purpose :
#
#* Creation Date : 10-06-2024
#
#* Last Modified : Mon 10 Jun 2024 02:40:22 AM IST
#
#* Created By : Yaay Nands
#_._._._._._._._._._._._._._._._._._._._._.#
import aec_rps

env = aec_rps.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()
