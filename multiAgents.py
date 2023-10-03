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
from typing import List, Type

from pacman import GameState
from util import manhattanDistance
from game import Directions, AgentState, GameStateData
import random, util
from game import Agent

PAC_MAN = 0
GHOSTS_BASE_INDEX = 1
BASE_STATE_EVAL = 50

# ------------------------------------------------------------------------ CLASS ---------------------------------------

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
        newFood = GameState(successorGameState).getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

# ------------------------------------------------------------------------ CLASS ---------------------------------------

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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

# ------------------------------------------------------------------------ CLASS ---------------------------------------



class MinimaxAgent(MultiAgentSearchAgent):
    """ ================================================================================================================
    ====================== MINIMAX AGENT ======================================================= QUESTION 2 ============
    ====================================================================================================================
    """

    def __init__(self, depth='2'):
        super().__init__()

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent where:
            - agentIndex=0 is Pacman
            - ghosts' agentIndex >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether the game state is a winning state

        gameState.isLose():
        Returns whether the game state is a losing state
        """
        return self.maxValue(gameState, self.depth)[1]


    # ------------------------------------------------------------------------------ MAX VALUE -------------------------
    def maxValue(self, thisGameState: GameState, depth: int) -> (float, any):
        """ Maximizes for the PacMan agent """
        if depth == 0:
            return theEvaluationOf(thisGameState), None

        highestScore, selectedAction = 0, None
        actions = thisGameState.getLegalActions(PAC_MAN)
        print("PACMAN Actions: ")
        print(actions)
        for action in actions:
            minResultingTuple = self.minValue(thisGameState.generateSuccessor(PAC_MAN, action), depth - 1)
            # From the resulting tuple, only the value matters. The action is the one from PacMan
            currentScore, currentAction = minResultingTuple[0], action
            if currentScore > highestScore:
                highestScore, selectedAction = currentScore, currentAction
        return highestScore, selectedAction

    # ------------------------------------------------------------------------------ MIN VALUE -------------------------
    def minValue(self, thisGameState: GameState, depth: int) -> (float, any):
        """ Minimizes for the Ghost agents """
        if depth == 0:
            return theEvaluationOf(thisGameState), None

        lowestScore, selectedAction = 9999999, None
        successorStates = self.getGhostsSuccessorStates(thisGameState)
        for successorState in successorStates:
            currentScore, currentAction = self.maxValue(successorState, depth - 1)
            if currentScore < lowestScore:
                lowestScore, selectedAction = currentScore, currentAction
        return lowestScore, selectedAction

    # ----------------------------------------------------------------------- GHOST SUCCESSORS STATES ------------------
    def getGhostsSuccessorStates(self, state: GameState)-> list[GameState]:
        """ Obtains all the possible successor states based on the possible combinations of the available ghosts'
            possible actions. Each successor state corresponds to one tuple whose actions were successively applied for
            its corresponding ghost.
        """
        totalGhosts = state.getNumAgents() - 1
        ghostActions = []
        for index in range(totalGhosts):
            ghostActions.append(state.getLegalActions(index))

        successorStates = []
        transitionTuples = self.generateStateTransitionTuples(ghostActions, totalGhosts)
        for transitionTuple in transitionTuples:
            ghostIndex = 1
            successor = state
            for ghostAction in transitionTuple:
                successor = successor.generateSuccessor(ghostIndex, ghostAction)
                ghostIndex += 1
            successorStates.append(successor)
        return successorStates

    # ----------------------------------------------------------------------- ALL COMBINATIONS OF GHOST ACTIONS --------
    def generateStateTransitionTuples(self, ghostsActions: list[list[any]], tupleLength: int) -> list[tuple]:
        """
        Builds a list of tuples where each tuple is a possible combinations of ghost legal actions.
        For ex:
                ghost_01_actions = {A, B}
                ghost_02_actions = {C, D, E}
                tuples = {(A, C), (A, D), (A, E), (B, C), (B, D), (B, E)}
            Args:
                ghostsActions a list L of lists Ln where each inner list Ln is a list of a ghost actions.
                tupleLength the tuple length. For any n-upla it's initial value is n.
            Return:
                A list of tuples where each tuple is combinations of ghost legal actions. Each tuple position index
                corresponds to the index of the ghost executing the action.
        """
        if tupleLength == 0:
            return [tuple()]

        if not ghostsActions:
            return []

        first_list = ghostsActions[0]
        rest_lists = ghostsActions[1:]

        combinations = []
        for element in first_list:
            for sub_combination in self.generateStateTransitionTuples(rest_lists, tupleLength - 1):
                combinations.append((element,) + sub_combination)

        print("GHOST Actions. State Transition tuples: ")
        print(combinations)
        return combinations

# ------------------------------------------------------------------------ CLASS ---------------------------------------

class AlphaBetaAgent(MultiAgentSearchAgent):
    """ ================================================================================================================
    ==================================== ALPHA-BETA PRUNING ==================================== QUESTION 3 ============
    ====================================================================================================================
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

# ------------------------------------------------------------------------ CLASS ---------------------------------------

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
        util.raiseNotDefined()

# ------------------------------------------------------------------------ CLASS ---------------------------------------
def betterEvaluationFunction(state: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Starts with a base value and either penalizes it (subtracts a certain amount) or rewards it (adds a
    certain amount). The more beneficial the scenario, the more we reward, the more detrimental, the more we penalize.
    Both win (super rewarding) and lose (super detrimental) final states are absolute on their own and will trigger an
    immediate return.
    """
    stateEvaluation = BASE_STATE_EVAL
    if state.isLose():
        return 0
    elif state.isWin():
        return stateEvaluation * 2
    else:
        return state.getScore()/state.getNumAgents() #+ rewardFunction(state)

def rewardFunction(state: GameState) -> float:
    stateEvaluation = BASE_STATE_EVAL

    # To chase ghosts or to run away from ghosts?
    ghostStates = list[AgentState](state.getGhostStates())
    pacManPos = state.getPacmanPosition()
    for ghostState in ghostStates:
        ghostPosition = ghostState.getPosition()
        distancePacManGhost = manhattanDistance(pacManPos, ghostPosition)
        if ghostState.scaredTimer:
            stateEvaluation = stateEvaluation - 2*distancePacManGhost
        else:
            stateEvaluation = stateEvaluation + distancePacManGhost

    # The fewer ghosts, the better
    stateEvaluation = stateEvaluation/state.getNumAgents()

    # Any capsule? If yes, the closer, the better
    capsules = state.getCapsules()
    if not capsules:
        stateEvaluation = stateEvaluation * 5
    else:
        for capsule in capsules:
            stateEvaluation = stateEvaluation - manhattanDistance(state.getPacmanPosition(), capsule.getPosition())

    # the higher the score, the higher the reward
    stateEvaluation = stateEvaluation * state.getScore()

    # reward for eating the food
    stateEvaluation = stateEvaluation/GameState(state).getNumFood()

    return stateEvaluation

# Abbreviation
theEvaluationOf = betterEvaluationFunction
