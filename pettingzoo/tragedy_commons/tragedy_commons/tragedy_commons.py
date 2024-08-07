# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : new_env.py
#
#* Purpose :
#
#* Creation Date : 10-06-2024
#
#* Last Modified : Mon 17 Jun 2024 11:58:42 PM IST
#
#* Created By : Yaay Nands
#_._._._._._._._._._._._._._._._._._._._._.#
import axelrod as axl
import functools
import gymnasium
import itertools as it
import numpy as np
import pygame
import os.path as path
import random

from statistics import mean
from easydict import EasyDict as edict
from gymnasium import spaces

from pettingzoo.classic import chess
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

# Local/custom imports
import board

COOPERATE = 0
DEFECT = 1
NONE = 2
MOVES = {COOPERATE: "COOPERATE",
         DEFECT: "DEFECT",
         NONE: "NONE"
         }
NUM_ITERS = 10
NUM_AGENTS = 10

def rev_lookup_moves(move):
    keys = list()
    for mv in move:
        keys.append(next(key for key, value in MOVES.items() if value == mv))
    return tuple(keys)

def reward_mapper_func(iterable):
    from collections import Counter
    cntr = Counter(iterable)
    return cntr.most_common()

def global_cost_and_average(reward_val):
    #TODO: There's a lot more nuances to be explored here
    #   For ex: Should we reward everyone equally or separately
    #            Should we change the global penalty based on which action is max CO-OPERATE or DEFECT or NONE
    global_res = 0
    rewards = edict(dict(reward_val))
    res = rewards.get('COOPERATE', 0)/len(MOVES)
    if 'DEFECT' in rewards.keys():
        global_res = -1 * res/(NUM_ITERS*NUM_AGENTS)
    return {'reward': res,
            'global': global_res
            }
def blind_average(reward_val):
    # Blind average reward returns to everyone without discrimination/punishpment for bad behaviour
    return mean([ea[1] for ea in reward_val])

def blind_max_voter(reward_val, moves,agent_idx):
    # reward_val --> counter of the moves
    # moves --> current moves list(ordered by agent/player)
    # agent_idx --> current agents' index
    # Returns the max vote value as reward if current agent got max votes
    # else returns  no reward -- so majority action takers will always win
    if moves.index(reward_val[0][0]) == agent_idx:
        return reward_val[0][1]
    return 0

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = TragedyCommonsEnv(render_mode=internal_render_mode, 
                            summarizer_func=global_cost_and_average)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class TragedyCommonsEnv(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
                "render_modes": ["human", "rgb_array"],
                "name": "tragedy_commons",
                "render_fps": 25
                }

    def __init__(self, render_mode=None, num_agents=NUM_AGENTS, screen_height=1000,
                                            summarizer_func=mean):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        print(axl.strategies)
        self.possible_agents = {"player_" + str(r): axl.strategies[0]()
                                        for r in range(num_agents) }
        self.summarizer_func = summarizer_func
        self.board = board.Board()

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        self._action_spaces = {agent: spaces.Discrete(len(MOVES)) for agent in self.possible_agents}
        #self._observation_spaces = {
        #    agent: spaces.Discrete(len(MOVES)) for agent in self.possible_agents
        #}

        self.agents = list(self.possible_agents.keys())
        self.observation_spaces = {
            name: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(8, 8), dtype=bool
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(4672,), dtype=np.int8
                    ),
                }
            )
            for name in self.agents
        }
        self.screen_height = self.screen_width = screen_height
        self.screen = None
        self.idx = (agent for agent in self.possible_agents.keys())
        self.possible_moves = list(it.product(MOVES.values(), repeat=NUM_AGENTS))
        self.reward_map = {rev_lookup_moves(move): reward_mapper_func(move) for move in self.possible_moves}
        self.render_mode = render_mode
        if self.render_mode in ["human", "rgb_array"]:
            self.BOARD_SIZE = (self.screen_width, self.screen_height)
            self.clock = pygame.time.Clock()
            self.cell_size = (self.BOARD_SIZE[0] / 8, self.BOARD_SIZE[1] / 8)

            bg_name = path.join(path.dirname(__file__), "img/ecosystem.png")
            self.bg_image = pygame.transform.scale(
                pygame.image.load(bg_name), self.BOARD_SIZE
            )

    def _accumulate_rewards(self):
        for agent,reward in self.rewards.items():
            if isinstance(reward, dict):
                self._cumulative_rewards[agent] += reward['reward']
            else:
                self._cumulative_rewards[agent] += reward


        pass
    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return spaces.Discrete(4)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(3)

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif self.render_mode == "ansi":
            return str(self.board)
        elif self.render_mode in {"human", "rgb_array"}:
            return self._render_gui()
        else:
            raise ValueError(
                f"{self.render_mode} is not a valid render mode. Available modes are: {self.metadata['render_modes']}"
            )

    def _render_gui(self):
        if self.screen is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.set_caption("Chess")
                self.screen = pygame.display.set_mode(self.BOARD_SIZE)
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface(self.BOARD_SIZE)

        bg_name = path.join(path.dirname(__file__), "img/ecosystem.png")
        self.bg_image = pygame.transform.scale(
            pygame.image.load(bg_name), self.BOARD_SIZE
        )
        self.screen.blit(self.bg_image, (0, 0))

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the otherbu
        #return np.array(self.observations[agent])
        return self.board.resources

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = list(self.possible_agents.keys())
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = edict({agent: NONE for agent in self.agents})
        self.observations = {agent: NONE for agent in self.agents}
        self.num_moves = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            reward_idx =  tuple([self.state.get(agent) for agent in self.agents])
            self.rewards[agent] = self.summarizer_func(self.reward_map[reward_idx])

            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.num_moves >= NUM_ITERS for agent in self.agents
            }

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.state[
                    self.agents[1 - self.agent_name_mapping[i]]
                ]

            self.board.update_resources(self.rewards)
            self.screen.blit(self.bg_image, (0, 0))
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

        if self.render_mode == "rgb_array":
            self.render()
#To interact with your custom AEC environment, use the following code:
