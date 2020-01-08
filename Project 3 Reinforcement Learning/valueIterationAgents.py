# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
from learningAgents import ValueEstimationAgent

import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            copy_values = self.values.copy()
            states = self.mdp.getStates()
            for state in states:
                self.values[state] = -999
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    value = 0
                    next_state_prob = self.mdp.getTransitionStatesAndProbs(state, action)
                    for Transition in next_state_prob:
                        next_state = Transition[0]
                        prob = Transition[1]
                        value += prob * (
                                self.mdp.getReward(state, action, next_state) + self.discount * copy_values[next_state])
                    if self.values[state] < value:
                        self.values[state] = value
                if self.values[state] == -999:
                    self.values[state] = 0

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0
        next_state_prob = self.mdp.getTransitionStatesAndProbs(state, action)
        for Transition in next_state_prob:
            next_state = Transition[0]
            prob = Transition[1]
            value += prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])
        return value
        util.raiseNotDefined()


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        max = -999
        actn = None
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            value = 0
            next_state_prob = self.mdp.getTransitionStatesAndProbs(state, action)
            for Transition in next_state_prob:
                next_state = Transition[0]
                prob = Transition[1]
                value += prob * (
                            self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])
            if max < value:
                max = value
                actn = action
        return actn
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            copy_values = self.values.copy()
            states = self.mdp.getStates()
            length = i%len(states)
            self.values[states[length]] = -999
            actions = self.mdp.getPossibleActions(states[length])
            for action in actions:
                value = 0
                next_state_prob = self.mdp.getTransitionStatesAndProbs(states[length], action)
                for Transition in next_state_prob:
                    next_state = Transition[0]
                    prob = Transition[1]
                    value += prob * (self.mdp.getReward(states[length], action, next_state) + self.discount * copy_values[next_state])
                if self.values[states[length]] < value:
                    self.values[states[length]] = value
            if self.values[states[length]] == -999:
                self.values[states[length]] = 0

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):

        predecessors = {}
        states = self.mdp.getStates()

        for state in states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                next_state_prob = self.mdp.getTransitionStatesAndProbs(state, action)
                for Transition in next_state_prob:
                    next_state = Transition[0]
                    prob = Transition[1]
                    if prob > 0:
                        if next_state in predecessors:
                            predecessors[next_state].append(state)
                        else:
                            predecessors[next_state] = [state]
        for state in states:
            predecessors[state] = set(predecessors[state])

        prty_q = util.PriorityQueue()
        for state in states:
            max_Q_value = -999
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                if max_Q_value < self.computeQValueFromValues(state,action):
                    max_Q_value = self.computeQValueFromValues(state,action)
            if max_Q_value != -999:
                diff = -abs(self.values[state] - max_Q_value)
                prty_q.push(state,diff)

        for i in range(self.iterations):
            if prty_q.isEmpty():
                return
            state = prty_q.pop()
            max_Q_value = -999
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                if max_Q_value < self.computeQValueFromValues(state,action):
                    max_Q_value = self.computeQValueFromValues(state,action)
            self.values[state] = max_Q_value

            for pred in predecessors[state]:
                max_Q_value = -999
                actions = self.mdp.getPossibleActions(pred)
                for action in actions:
                    if max_Q_value < self.computeQValueFromValues(pred, action):
                        max_Q_value = self.computeQValueFromValues(pred, action)
                if max_Q_value != -999:
                    diff = -abs(self.values[pred] - max_Q_value)
                    if (-diff) > self.theta:
                        prty_q.update(pred, diff)

        "*** YOUR CODE HERE ***"

