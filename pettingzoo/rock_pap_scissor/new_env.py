# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : new_env.py
#
#* Purpose :
#
#* Creation Date : 10-06-2024
#
#* Last Modified : Mon 10 Jun 2024 02:36:40 AM IST
#
#* Created By : Yaay Nands
#_._._._._._._._._._._._._._._._._._._._._.#
import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

ROCK = 0
PAPER = 1
SCISSORS = 2
NONE = 3
MOVES = ["ROCK", "PAPER", "SCISSORS", "None"]
NUM_ITERS = 100
REWARD_MAP = {
    (ROCK, ROCK): (0, 0),
    (ROCK, PAPER): (-1, 1),
    (ROCK, SCISSORS): (1, -1),
    (PAPER, ROCK): (1, -1),
    (PAPER, PAPER): (0, 0),
    (PAPER, SCISSORS): (-1, 1),
    (SCISSORS, ROCK): (-1, 1),
    (SCISSORS, PAPER): (1, -1),
    (SCISSORS, SCISSORS): (0, 0),
}


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None):
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
        self.possible_agents = ["player_" + str(r) for r in range(2)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        self._action_spaces = {agent: Discrete(3) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Discrete(4) for agent in self.possible_agents
        }
        self.render_mode = render_mode

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Discrete(4)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == 2:
            string = "Current state: Agent1: {} , Agent2: {}".format(
                MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]
            )
        else:
            string = "Game over"
        print(string)

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
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
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: NONE for agent in self.agents}
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
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[
                (self.state[self.agents[0]], self.state[self.agents[1]])
            ]

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

#To interact with your custom AEC environment, use the following code:
#
#import aec_rps
#
#env = aec_rps.env(render_mode="human")
#env.reset(seed=42)
#
#for agent in env.agent_iter():
#    observation, reward, termination, truncation, info = env.last()
#
#    if termination or truncation:
#        action = None
#    else:
#        # this is where you would insert your policy
#        action = env.action_space(agent).sample()
#
#    env.step(action)
#env.close()
#
#Example Custom Parallel Environment
#
#import functools
#
#import gymnasium
#from gymnasium.spaces import Discrete
#
#from pettingzoo import ParallelEnv
#from pettingzoo.utils import parallel_to_aec, wrappers
#
#ROCK = 0
#PAPER = 1
#SCISSORS = 2
#NONE = 3
#MOVES = ["ROCK", "PAPER", "SCISSORS", "None"]
#NUM_ITERS = 100
#REWARD_MAP = {
#    (ROCK, ROCK): (0, 0),
#    (ROCK, PAPER): (-1, 1),
#    (ROCK, SCISSORS): (1, -1),
#    (PAPER, ROCK): (1, -1),
#    (PAPER, PAPER): (0, 0),
#    (PAPER, SCISSORS): (-1, 1),
#    (SCISSORS, ROCK): (-1, 1),
#    (SCISSORS, PAPER): (1, -1),
#    (SCISSORS, SCISSORS): (0, 0),
#}
#
#
#def env(render_mode=None):
#    """
#    The env function often wraps the environment in wrappers by default.
#    You can find full documentation for these methods
#    elsewhere in the developer documentation.
#    """
#    internal_render_mode = render_mode if render_mode != "ansi" else "human"
#    env = raw_env(render_mode=internal_render_mode)
#    # This wrapper is only for environments which print results to the terminal
#    if render_mode == "ansi":
#        env = wrappers.CaptureStdoutWrapper(env)
#    # this wrapper helps error handling for discrete action spaces
#    env = wrappers.AssertOutOfBoundsWrapper(env)
#    # Provides a wide vareity of helpful user errors
#    # Strongly recommended
#    env = wrappers.OrderEnforcingWrapper(env)
#    return env
#
#
#def raw_env(render_mode=None):
#    """
#    To support the AEC API, the raw_env() function just uses the from_parallel
#    function to convert from a ParallelEnv to an AEC env
#    """
#    env = parallel_env(render_mode=render_mode)
#    env = parallel_to_aec(env)
#    return env
#
#
#class parallel_env(ParallelEnv):
#    metadata = {"render_modes": ["human"], "name": "rps_v2"}
#
#    def __init__(self, render_mode=None):
#        """
#        The init method takes in environment arguments and should define the following attributes:
#        - possible_agents
#        - render_mode
#
#        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
#        Spaces should be defined in the action_space() and observation_space() methods.
#        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.
#
#        These attributes should not be changed after initialization.
#        """
#        self.possible_agents = ["player_" + str(r) for r in range(2)]
#
#        # optional: a mapping between agent name and ID
#        self.agent_name_mapping = dict(
#            zip(self.possible_agents, list(range(len(self.possible_agents))))
#        )
#        self.render_mode = render_mode
#
#    # Observation space should be defined here.
#    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
#    # If your spaces change over time, remove this line (disable caching).
#    @functools.lru_cache(maxsize=None)
#    def observation_space(self, agent):
#        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
#        return Discrete(4)
#
#    # Action space should be defined here.
#    # If your spaces change over time, remove this line (disable caching).
#    @functools.lru_cache(maxsize=None)
#    def action_space(self, agent):
#        return Discrete(3)
#
#    def render(self):
#        """
#        Renders the environment. In human mode, it can print to terminal, open
#        up a graphical window, or open up some other display that a human can see and understand.
#        """
#        if self.render_mode is None:
#            gymnasium.logger.warn(
#                "You are calling render method without specifying any render mode."
#            )
#            return
#
#        if len(self.agents) == 2:
#            string = "Current state: Agent1: {} , Agent2: {}".format(
#                MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]
#            )
#        else:
#            string = "Game over"
#        print(string)
#
#    def close(self):
#        """
#        Close should release any graphical displays, subprocesses, network connections
#        or any other environment data which should not be kept around after the
#        user is no longer using the environment.
#        """
#        pass
#
#    def reset(self, seed=None, options=None):
#        """
#        Reset needs to initialize the `agents` attribute and must set up the
#        environment so that render(), and step() can be called without issues.
#        Here it initializes the `num_moves` variable which counts the number of
#        hands that are played.
#        Returns the observations for each agent
#        """
#        self.agents = self.possible_agents[:]
#        self.num_moves = 0
#        observations = {agent: NONE for agent in self.agents}
#        infos = {agent: {} for agent in self.agents}
#        self.state = observations
#
#        return observations, infos
#
#    def step(self, actions):
#        """
#        step(action) takes in an action for each agent and should return the
#        - observations
#        - rewards
#        - terminations
#        - truncations
#        - infos
#        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
#        """
#        # If a user passes in actions with no agents, then just return empty observations, etc.
#        if not actions:
#            self.agents = []
#            return {}, {}, {}, {}, {}
#
#        # rewards for all agents are placed in the rewards dictionary to be returned
#        rewards = {}
#        rewards[self.agents[0]], rewards[self.agents[1]] = REWARD_MAP[
#            (actions[self.agents[0]], actions[self.agents[1]])
#        ]
#
#        terminations = {agent: False for agent in self.agents}
#
#        self.num_moves += 1
#        env_truncation = self.num_moves >= NUM_ITERS
#        truncations = {agent: env_truncation for agent in self.agents}
#
#        # current observation is just the other player's most recent action
#        observations = {
#            self.agents[i]: int(actions[self.agents[1 - i]])
#            for i in range(len(self.agents))
#        }
#        self.state = observations
#
#        # typically there won't be any information in the infos, but there must
#        # still be an entry for each agent
#        infos = {agent: {} for agent in self.agents}
#
#        if env_truncation:
#            self.agents = []
#
#        if self.render_mode == "human":
#            self.render()
#        return observations, rewards, terminations, truncations, infos
#
##To interact with your custom parallel environment, use the following code:
#
##import parallel_rps
##
##env = parallel_rps.parallel_env(render_mode="human")
##observations, infos = env.reset()
##
##while env.agents:
##    # this is where you would insert your policy
##    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
##
##    observations, rewards, terminations, truncations, infos = env.step(actions)
##env.close()
#
#
