# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        print(successorGameState.getScore())

        "*** YOUR CODE HERE ***"
        food_points = newFood.asList()
        food_distances = []

        for food in food_points:
            food_distances.append(1/util.manhattanDistance(food, newPos))
        return successorGameState.getScore() + sum(food_distances)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def max_v(gameState, depth, agent):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), Directions.STOP

            v = float("-inf")
            succ_action = Directions.STOP
            actions = gameState.getLegalActions()
            for action in actions:
                successor = gameState.generateSuccessor(agent,action)
                successorValue = min_v(successor, depth, 1)[0]
                if (successorValue > v):
                    v = successorValue
                    succ_action = action
            return v, succ_action

        def min_v(gameState, depth, agent):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), Directions.STOP

            v = float("inf")
            succ_action = Directions.STOP
            actions = gameState.getLegalActions(agent)
            agents = gameState.getNumAgents()
            for action in actions:
                successor = gameState.generateSuccessor(agent, action)
                if (agent == agents - 1):
                    successorValue = max_v(successor, depth + 1, 0)[0]
                else:
                    successorValue = min_v(successor, depth, agent + 1,)[0]

                if (successorValue < v):
                    v= successorValue
                    succ_action = action
            return v, succ_action

        depth = 0
        agent = 0
        return max_v(gameState, depth, agent)[1]


        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def max_v(gameState, depth, agent, alpha,beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), Directions.STOP

            v = float("-inf")
            succ_action = Directions.STOP
            actions = gameState.getLegalActions()
            for action in actions:
                successor = gameState.generateSuccessor(agent,action)
                successorValue = min_v(successor, depth, 1, alpha, beta)[0]
                if (successorValue > v):
                    v = successorValue
                    succ_action = action
                if successorValue > beta:
                    return v, succ_action
                if successorValue > alpha:
                    alpha = successorValue
            return v, succ_action

        def min_v(gameState, depth, agent, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), Directions.STOP

            v = float("inf")
            succ_action = Directions.STOP
            actions = gameState.getLegalActions(agent)
            agents = gameState.getNumAgents()
            for action in actions:
                successor = gameState.generateSuccessor(agent, action)
                if (agent == agents - 1):
                    successorValue = max_v(successor, depth + 1, 0, alpha, beta)[0]
                else:
                    successorValue = min_v(successor, depth, agent + 1, alpha, beta)[0]

                if (successorValue < v):
                    v= successorValue
                    succ_action = action

                if successorValue < alpha:
                    return v, succ_action
                if successorValue < beta:
                    beta = successorValue
            return v, succ_action

        depth = 0
        agent = 0
        alpha = float("-inf")
        beta = float("inf")
        return max_v(gameState, depth, agent, alpha, beta)[1]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_v(gameState, depth, agent, alpha,beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), Directions.STOP

            v = float("-inf")
            succ_action = Directions.STOP
            actions = gameState.getLegalActions()
            for action in actions:
                successor = gameState.generateSuccessor(agent,action)
                successorValue = min_v(successor, depth, 1, alpha, beta)[0]
                if (successorValue > v):
                    v = successorValue
                    succ_action = action
                if successorValue > beta:
                    return v, succ_action
                if successorValue > alpha:
                    alpha = successorValue
            return v, succ_action

        def min_v(gameState, depth, agent, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), Directions.STOP

            v = float("inf")
            succ_action = Directions.STOP
            actions = gameState.getLegalActions(agent)
            agents = gameState.getNumAgents()
            for action in actions:
                successor = gameState.generateSuccessor(agent, action)
                if (agent == agents - 1):
                    successorValue = max_v(successor, depth + 1, 0, alpha, beta)[0]
                else:
                    successorValue = min_v(successor, depth, agent + 1, alpha, beta)[0]

                if (successorValue < v):
                    v= successorValue
                    succ_action = action

                if successorValue < alpha:
                    return v, succ_action
                if successorValue < beta:
                    beta = successorValue
            return v, succ_action

        depth = 0
        agent = 0
        alpha = float("-inf")
        beta = float("inf")
        return max_v(gameState, depth, agent, alpha, beta)[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    Position = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

    score2 = 0
    score3 = 0
    food_points = Food.asList()
    distances = 0

    for food in food_points:
        distances += 1 / util.manhattanDistance(food, Position)

    for ghost in GhostStates:
        ghostpos = ghost.getPosition()
        ghostDist = util.manhattanDistance(ghostpos, Position)
        if ghostDist > 1:
            score2 = score2 - (1.0 / ghostDist)
        if ghost.scaredTimer > 0:
            score3 = score3 + (10.0 / ghostDist)

    return currentGameState.getScore() + distances + score2 + score3

# Abbreviation
better = betterEvaluationFunction
