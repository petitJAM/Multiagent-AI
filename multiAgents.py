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

import pacman

import pprint
pp = pprint.PrettyPrinter(indent=2)

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
    currentFood = currentGameState.getFood().asList()

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

    if action == Directions.STOP: # don't stop
      return -999999

    foodList = newFood.asList()

    distancesFromFoods = [manhattanDistance(fp, newPos) for fp in foodList]
    minFoodDist = min(distancesFromFoods) if len(distancesFromFoods) > 0 else 1

    newGhostPos = [ghostState.getPosition() for ghostState in newGhostStates]

    if newPos in currentFood:
      return 999999

    return -minFoodDist
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
    #####################################################
    def evalTreeKey(x):
      return x[0] if isinstance(x, tuple) else x

    def minimax(gameState, agentIndex, currentDepth):
      # print "Recursion at", currentDepth, "agent", agentIndex

      legalMoves = gameState.getLegalActions(agentIndex)

      try:
        legalMoves.remove(Directions.STOP) # make it faster
      except ValueError:
        pass

      nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
      nextDepth = currentDepth + (1 if nextAgentIndex == 0 else 0)

      if currentDepth > self.depth:
        return self.evaluationFunction(gameState), None

      evalList = []
      for m in legalMoves:
        successorGameState = gameState.generateSuccessor(agentIndex, m)
        evalList.append((minimax(successorGameState, nextAgentIndex, nextDepth)[0], m)) # the m in (_, m) is to track 
                                                                                        # the move pacman should take 
      try:                                                                              # at the top level of the recursion
        if agentIndex == 0:
          return max(evalList, key=evalTreeKey)
        else:
          return min(evalList, key=evalTreeKey)
      except ValueError: # empty
        return self.evaluationFunction(gameState), None
    #####################################################

    ret, m = minimax(gameState, 0, 1)

    # print ret, m

    return m


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"

    def evalTreeKey(x):
      return x[0] if isinstance(x, tuple) else x

    def alphabeta(gameState, agentIndex, currentDepth, alpha, beta):
      # print "Recursion at", currentDepth, "agent", agentIndex

      legalMoves = gameState.getLegalActions(agentIndex)

      nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
      nextDepth = currentDepth + (1 if nextAgentIndex == 0 else 0)

      if currentDepth > self.depth or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState), None

      if agentIndex == 0:
        alphastate = None
        for m in legalMoves:
          state = gameState.generateSuccessor(agentIndex, m)
          v = alphabeta(state, nextAgentIndex, nextDepth, alpha, beta)[0]
          if v > alpha and beta >= v: # new alpha and not going to prune
            alphastate = m
          alpha = max(alpha, v)
          if beta <= alpha:
            break
        return alpha, alphastate # only matters for top level nodes

      else:
        for m in legalMoves:
          state = gameState.generateSuccessor(agentIndex, m)
          v = alphabeta(state, nextAgentIndex, nextDepth, alpha, beta)[0]
          beta = min(beta, v)
          if beta <= alpha:
            break
        return beta, None # don't need to return states on ghosts

    _, m = alphabeta(gameState, 0, 1, -999999, 999999)

    # print _, m

    return m

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
    def evalTreeKey(x):
      return x[0] if isinstance(x, tuple) else x

    def expectimax(gameState, agentIndex, currentDepth):
      # print "Recursion at", currentDepth, "agent", agentIndex

      legalMoves = gameState.getLegalActions(agentIndex)

      try:
        legalMoves.remove(Directions.STOP) # make it faster
      except ValueError:
        pass

      nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
      nextDepth = currentDepth + (1 if nextAgentIndex == 0 else 0)

      if currentDepth > self.depth or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState), None

      if agentIndex == 0:
        alphastate = None
        alpha = -999999
        for m in legalMoves:
          state = gameState.generateSuccessor(agentIndex, m)
          v = expectimax(state, nextAgentIndex, nextDepth)[0]
          if v > alpha: # new alpha
            alpha = v
            alphastate = m
        return alpha, alphastate # only matters for top level nodes

      else:
        alpha = 0
        for m in legalMoves:
          state = gameState.generateSuccessor(agentIndex, m)
          alpha += expectimax(state, nextAgentIndex, nextDepth)[0] / len(legalMoves)
        return alpha, None

    _, m = expectimax(gameState, 0, 1)

    # print _, m

    return m































def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
      Win/Lose - Obviously winning is good and losing bad

      Go through all food and find the closest by manhattanDistance.
        Then using that food, create a PositionSearchProblem and find 
        the length of the path to it.

      Do the same for capsules


  """
  "*** YOUR CODE HERE ***"

  def pathToPoint(pointTo, pointFrom):
    posSearchProblem = PositionSearchProblem(currentGameState, lambda x: 1, pointTo, pointFrom, False)
    path = uniformCostSearch(posSearchProblem)
    return path


  if currentGameState.isWin():
    return 999999
  if currentGameState.isLose():
    return -999999

  # grab some info
  pacPos      = currentGameState.getPacmanPosition()

  foods       = currentGameState.getFood().asList()
  numFood     = currentGameState.getNumFood()
  capsules    = currentGameState.getCapsules()

  ghostStates = currentGameState.getGhostStates()
  scaredTimes = [ghs.scaredTimer for ghs in ghostStates]

  sorted(ghostStates, key=lambda gs: manhattanDistance(pacPos, gs.getPosition()))

  ##### ideas #####
  # find the closest X foods
  # include capsules and scared ghosts
  # probably just consider scared ghosts to be food.  worth more somehow?
  # if trapped, get more food if possible, then die quickly
  #################

  # closest food

  minFood = min([(fp, manhattanDistance(fp, pacPos)) for fp in foods], key=lambda x: x[1])[0]
  minFoodDist = manhattanDistance(minFood, pacPos)
  pathToFoodLength = len(pathToPoint(minFood, pacPos))

  # closest capsule

  if len(capsules) > 0:
    minCapsule = min([(cap, manhattanDistance(cap, pacPos)) for cap in capsules], key=lambda x: x[1])[0]
    minCapsuleDist = manhattanDistance(minCapsule, pacPos)
    pathToCapsuleLength = len(pathToPoint(minCapsule, pacPos))
  else:
    pathToCapsuleLength = 999999

  # scared ghosts

  scaredGhostVal = 1
  ateACapsule = 0

  for gs in ghostStates:
    if pacman.SCARED_TIME == gs.scaredTimer:
      ateACapsule = 3

    if pacman.SCARED_TIME - 2 > gs.scaredTimer > 0:
      # print "scared ghost"
      gpos = gs.getPosition()
      gpos = (int(gpos[0]), int(gpos[1]))
      path = pathToPoint(gpos, pacPos)
      pathLenToGhost = len(path)
      if pathLenToGhost < gs.scaredTimer * 2:
        scaredGhostVal = pathLenToGhost

  # sum up everything

  game_score_score   = currentGameState.getScore()
  min_food_score     = 1.0 / minFoodDist # manhattanDistance
  path_len_score     = 2.0 / pathToFoodLength # actual path
  min_capsule_score  = 3.0 / pathToCapsuleLength
  scared_ghost_score = 4.0 / scaredGhostVal

  final_score = game_score_score + min_food_score + path_len_score + min_capsule_score + scared_ghost_score + ateACapsule

  # print game_score_score, min_food_score, path_len_score, min_capsule_score, scared_ghost_score, final_score

  # raw_input()

  return final_score


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


class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

class PositionSearchProblem(SearchProblem):
  """
  A search problem defines the state space, start state, goal test,
  successor function and cost function.  This search problem can be 
  used to find paths to a particular point on the pacman board.
  
  The state space consists of (x,y) positions in a pacman game.
  
  Note: this search problem is fully specified; you should NOT change it.
  """
  
  def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True):
    """
    Stores the start and goal.  
    
    gameState: A GameState object (pacman.py)
    costFn: A function from a search state (tuple) to a non-negative number
    goal: A position in the gameState
    """
    self.walls = gameState.getWalls()
    self.startState = gameState.getPacmanPosition()
    if start != None: self.startState = start
    self.goal = goal
    self.costFn = costFn
    if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
      print 'Warning: this does not look like a regular search maze'

    # For display purposes
    self._visited, self._visitedlist, self._expanded = {}, [], 0

  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
     isGoal = state == self.goal 
     
     # For display purposes only
     if isGoal:
       self._visitedlist.append(state)
       
     return isGoal   
   
  def getSuccessors(self, state):
    """
    Returns successor states, the actions they require, and a cost of 1.
    
     As noted in search.py:
         For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
    """
    
    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state
      dx, dy = self.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextState = (nextx, nexty)
        cost = self.costFn(nextState)
        successors.append( ( nextState, action, cost) )
        
    # Bookkeeping for display purposes
    self._expanded += 1 
    if state not in self._visited:
      self._visited[state] = True
      self._visitedlist.append(state)
      
    return successors

  def directionToVector(self, action):
    if action == Directions.NORTH:
      return 0, 1
    elif action == Directions.SOUTH:
      return 0, -1
    elif action == Directions.WEST:
      return -1, 0
    elif action == Directions.EAST:
      return 1, 0

  def getCostOfActions(self, actions):
    """
    Returns the cost of a particular sequence of actions.  If those actions
    include an illegal move, return 999999
    """
    if actions == None: return 999999
    x,y= self.getStartState()
    cost = 0
    for action in actions:
      # Check figure out the next state and see whether its' legal
      dx, dy = self.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]: return 999999
      cost += self.costFn((x,y))
    return cost


def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  start_state = problem.getStartState()

  node_pq = util.PriorityQueue()

  visited_states = set([start_state])
  for next in problem.getSuccessors(start_state):
    node_pq.push([next], next[2])

  while not node_pq.isEmpty():
    path = node_pq.pop()

    # if path[-1][0] in visited_states:
    #   continue

    visited_states.add(path[-1][0])

    if problem.isGoalState(path[-1][0]):
      return [p[1] for p in path]

    for successor in problem.getSuccessors(path[-1][0]):
      if not successor[0] in visited_states:
        # next_path = list(path)
        # next_path.append(successor)
        # node_pq.push(next_path, problem.getCostOfActions([p[1] for p in path]))
        existing_priority, indexof = priorityQueueContains(node_pq, successor[0])
        cost = problem.getCostOfActions([p[1] for p in path] + [successor[1]])

        if not existing_priority:
          next_path = list(path)
          next_path.append(successor)
          node_pq.push(next_path, cost)
        elif cost < existing_priority:
          node_pq.heap[indexof] = (cost, node_pq.heap[indexof][1]) # change priority
          heapq.heapify(node_pq.heap)             # reset the pq

# returns (priority, index) if it exists
def priorityQueueContains(pq, item):
  for i, pair in enumerate(pq.heap):
    if pair[1][-1][0] == item: # my priority queue contains paths.  pair[1][-1][0] get the last state of those lists.
      return (pair[0], i)                                      # --> (priority, item) 
  return (False, None)                                                        # --> [(state, dir, cost), ...]

