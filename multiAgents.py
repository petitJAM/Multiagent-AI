# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]

    # Don't choose STOP if possible
    if len(bestIndices) > 1:
      for i in bestIndices:
        if legalMoves[i] == Directions.STOP:
          bestIndices.remove(i)

    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    # print "bestScore:", bestScore
    # print "bestIndices:", bestIndices, "chose:", chosenIndex
    # print "choosing from scores:", scores, "on", legalMoves

    # raw_input()
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
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()

    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action) # the state to evaluate
    newPos = successorGameState.getPacmanPosition() # (x,y)
    newFood = successorGameState.getFood() # Grid
    newGhostStates = successorGameState.getGhostStates() # something else?
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] # number of frames left of each ghost being scared

    if successorGameState.isLose():
      return -999999

    if successorGameState.isWin():
      return 999999

    capsules = currentGameState.getCapsules() # use current because if we hit one, it's gone in successor

    "*** YOUR CODE HERE ***"
    capsulesScore = 100 if newPos in capsules else 0

    foodList = newFood.asList()

    distancesFromFoods = []
    for fp in foodList:
      distancesFromFoods.append(manhattanDistance(fp, newPos))

    minFoodDist = min(distancesFromFoods) if len(distancesFromFoods) > 0 else 0
    # print minFoodDist, distancesFromFoods

    # dist from avg food position
    avgFoodDist = 0
    if len(foodList) > 0:
      avgFood = (sum([f[0] for f in foodList]) / len(foodList), sum([f[1] for f in foodList]) / len(foodList))
      avgFoodDist = manhattanDistance(avgFood, newPos)

    # close to new ghost pos
    newGhostPos = [ghostState.getPosition() for ghostState in newGhostStates]

    distancesFromGhosts = []
    for i, ghp in enumerate(newGhostPos):
      ghostDist = manhattanDistance(newPos, ghp)
      if newScaredTimes[i] == 0:
        distancesFromGhosts.append(ghostDist)
      else: # ignore ghost if it's scared
        distancesFromGhosts.append(100)

    if newPos in newGhostPos: # we hit one
      # print "newPos", newPos, "in", "newGhostPos"
      return -999999

    # if 0 in distancesFromGhosts: # we hit one
    #   return -999999

    # The base score is still the gamestate score
    # then the average distance from a ghost is added (farther away is better)
    # then the distance from the average food position
    # is subtracted (smaller is better)
    return successorGameState.getScore() + (1/sum(distancesFromGhosts) - 1/avgFoodDist + capsulesScore - minFoodDist) - (1 if action == Directions.STOP else 0)
    # return successorGameState.getScore() - 1/(avgGhostDist+1) + 10/(avgFoodDist+1) + capsulesScore

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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
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
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
