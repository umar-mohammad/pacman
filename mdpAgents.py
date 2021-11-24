# sampleAgents.py
# parsons/07-oct-2017
#
# Version 1.1
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

import api
import game
import util
import random

from game import Directions

class MDPAgent(game.Agent):

    def __init__(self):
        self.map = None
        self.mapValues = None

    def registerInitialState(self, state):
        """
        build initial map to be used for value iteration 
        """
        smallGridValues = {
            "empty" : 0,
            "ghost" : -6,
            "edible_ghost" : 30,
            "food" : 3,
            "pacman" : -5,
            "capsule" : 11,
            "wall" : "W"
        }

        mediumClassicValues = {
                "empty" : -1,
                "ghost" : -500,
                "edible_ghost" : 100,
                "food" : 10,
                "pacman" : -5,
                "capsule" : 15,
                "deathzone" : -25,
                "wall" : "W"
        }

        map = []

        max_x=0
        max_y=0

        for x,y in api.corners(state):
            max_x = max(x, max_x)
            max_y = max(y, max_y)
        
        if max_x == 19 and max_y == 10:
            # mediumClassic map
            self.mapValues = mediumClassicValues
        else:
            self.mapValues = smallGridValues

        for _ in range(max_x+1):
            map.append([0] * (max_y+1))
        
        for x, y in api.walls(state):
            map[x][y] = self.mapValues["wall"]

        self.map = map
        
    # This is what gets run in between multiple games
    def final(self, state):
        self.map = None

    def getAction(self, state):
        legalMoves = api.legalActions(state)
        val_it_map = self.valueIteration(self.getRewardMap(state), 0.999)
        max_move = maximisingAction(api.whereAmI(state), legalMoves, val_it_map)
        
        # print('\n' * 5)
        # printMap(val_it_map)
        # import time
        # time.sleep(3)

        return api.makeMove(max_move, legalMoves)

    def valueIteration(self, rewards, gamma=0.9):
        """
        returns the utility values for each coordinate of the map
        """
        map = self.map # copies the values from the previous time step, which allows the algorithm to converge faster
        # iterations = 0
        while True:
            # iterations+=1
            oldMap = copyMap(map)
            delta = 0
            for x in range(len(map)):
                for y in range(len(map[0])):
                    reward = rewards[x][y]
                    if reward == "W":
                        map[x][y] = "W"
                    else:
                        newVal = rewards[x][y] + gamma * weightedExpectedUtility((x,y), getLegalActions((x,y), oldMap), oldMap, 0.9, 0.1)
                        delta += abs(newVal - oldMap[x][y])
                        map[x][y] = newVal
            if delta <= 5: 
                # print(iterations)
                break
        # self.iterations += iterations
        # self.n_vit +=1
        # self.max_iterations = max(iterations, self.max_iterations)
        self.map = map
        return map
    
    def getRewardMap(self, state):
        """
        return a 2D list representing the game state with reward values
        """
        # improvement: could make so that the map only checks a few cells in the game grid 
        # and update instead of rebuilding the whole map over and over

        # improvement: could store and reuse the map array from previous iterations 
        # so that we dont have to rebuild base array each iteration
        map = []

        pacman_x, pacman_y = api.whereAmI(state)
        walls = api.walls(state)
        corners = api.corners(state)
        ghosts = api.ghosts(state)
        edible_ghost = dict(api.ghostStatesWithTimes(state))
        food = api.food(state)
        capsules = api.capsules(state)

        max_x = 0
        max_y = 0

        for x,y in corners:
            max_x = max(x, max_x)
            max_y = max(y, max_y)

        # build initial map with empty spaces for every position in grid
        for _ in range(max_x+1):
            map.append([self.mapValues["empty"]] * (max_y+1))

        # place all game elements to the map
        map[pacman_x][pacman_y] = self.mapValues["pacman"]
        
        for x,y in capsules:
            map[x][y] = self.mapValues["capsule"]

        for x,y in food:
            map[x][y] = self.mapValues["food"]
        
        for x, y in walls:
            map[x][y] = self.mapValues["wall"]
        
        for pos in ghosts:
            # bug : ghosts position can be like x.5, y.5 but pacman's map array does not have indeces such as 1.5 
            # therefore pacman's internal representation of where the ghosts are might be slightly wrong
            # and this could lead to bugs/inaccuracies
            x,y = util.nearestPoint(pos)
            value = ((self.mapValues["edible_ghost"] * edible_ghost[pos])/40) if pos in edible_ghost and edible_ghost[pos] > 0 else self.mapValues["ghost"]

            map[x][y] += value

            for futureposition, steps in ghostFuture(pos, 4, map):
                x,y = futureposition
                map[x][y] += value * (1/steps)
            
        if max_x == 19 and max_y == 10:
            # mediumClassic map
                for x in range(8,12,1):
                    map[x][5] += self.mapValues["deathzone"]


        return map

def getLegalActions(position, map):
    """
    returns all legal moves for a particular position 
    """
    moves=[(0,1), (1, 0), (0, -1), (-1, 0)]
    legalActions=[Directions.STOP]
    wall = "W"
    for i in range(len(moves)):
        x,y = (moves[i][0] + position[0], moves[i][1] + position[1])
        if map[x][y] != wall:
            if i == 0: legalActions.append(Directions.NORTH)
            elif i == 1: legalActions.append(Directions.EAST)
            elif i == 2: legalActions.append(Directions.SOUTH)
            elif i == 3: legalActions.append(Directions.WEST) 
    return legalActions

def ghostFuture(position, steps, map):
    """
    returns the position where the ghost can be in future time steps
    """
    x,y = util.nearestPoint(position)
    futurePositions = []
    futurePositions.extend(ghostFutureHelper((x,y), Directions.NORTH, 0, steps, map))
    futurePositions.extend(ghostFutureHelper((x,y), Directions.EAST, 0, steps, map))
    futurePositions.extend(ghostFutureHelper((x,y), Directions.SOUTH, 0, steps, map))
    futurePositions.extend(ghostFutureHelper((x,y), Directions.WEST, 0, steps, map))

    return futurePositions

def ghostFutureHelper(position, direction, stepstaken, steps, map):
    wall = "W"
    x, y = util.nearestPoint(position)
    futurePositions = []
    x_add, y_add  = {
        Directions.NORTH : (0,1),
        Directions.EAST : (1,0),
        Directions.SOUTH : (0,-1),
        Directions.WEST : (-1,0)
    }[direction]

    for i in range(1, (steps-stepstaken)+1, 1):
        if map[x+(i*x_add)][y+(i*y_add)] != wall:
            futurePositions.append(((x+x_add, y+y_add), stepstaken+i))
            futurePositions.extend(ghostFutureHelper((x+(i*x_add), y+(i*y_add)), Directions.LEFT[direction], stepstaken+i, steps, map))
            futurePositions.extend(ghostFutureHelper((x+(i*x_add), y+(i*y_add)), Directions.RIGHT[direction], stepstaken+i, steps, map))
        else:
            break
    
    return futurePositions


def copyMap(map):
    """
    returns a copy of the map provided
    """
    copy = []

    for x in range(len(map)):
        copy.append([0] * (len(map[0])))
    
    for x in range(len(map)):
        for y in range(len(map[0])):
            copy[x][y] = map[x][y]
    
    return copy

def weightedExpectedUtility(position, legalMoves, rewardMap, positiveWeight=0.85, negativeWeight=0.15):
    """
    calculates a weighted expected utility for a certain position on the map, taking into account both
    the maximum and minimum reward you can get for a cell, this helps adjust how optimistic pacman is
    """
    bestMoveUtility = 0
    worstMoveUtility = 0

    for move in legalMoves:
        x,y = coordinateAfterMove(position, move, legalMoves)
        utility_move = api.directionProb * rewardMap[x][y]

        # calculate the rewards if the move does not execute like we want to
        error_moves = errorMoves(move)
        for error_move in error_moves:
            x,y = coordinateAfterMove(position, error_move, legalMoves)
            utility_move += ((1-api.directionProb)/len(error_moves)) * rewardMap[x][y]
        
        bestMoveUtility = max(bestMoveUtility, utility_move)
        worstMoveUtility = min(worstMoveUtility, utility_move)
    
    return positiveWeight * bestMoveUtility + negativeWeight * worstMoveUtility


def minimumExpectedUtility(position, legalMoves, rewardMap):
    """
    calculates minimum expected utility for a certain position on the map
    """
    worstMove = []

    for move in legalMoves:
        x,y = coordinateAfterMove(position, move, legalMoves)
        utility_move = api.directionProb * rewardMap[x][y]

        # calculate the rewards if the move does not execute like we want to
        error_moves = errorMoves(move)
        for error_move in error_moves:
            x,y = coordinateAfterMove(position, error_move, legalMoves)
            utility_move += ((1-api.directionProb)/len(error_moves)) * rewardMap[x][y]
        
        if len(worstMove) == 0 or utility_move < worstMove[0][1]:
            worstMove = [(move, utility_move)]
        elif utility_move == worstMove[0][1]:
            worstMove.append((move, utility_move))
        
    return random.choice(worstMove)[1]
        
def calculateBestActions(position, legalMoves, rewardMap):
    """
    calculates the best actions based on the rewards for a particular
    pacman position
    """
    # maximisingMoves is a list containing all move-utility tuples that have the highest utility value
    # this allows us to choose a random move which makes pacman's movement non-deterministic 
    # when multiple moves have equal utility value which helps it not get stuck in corners
    maximisingMoves = [] 

    for move in legalMoves:
        x,y = coordinateAfterMove(position, move, legalMoves)
        utility_move = api.directionProb * rewardMap[x][y]

        # calculate the rewards if the move does not execute like we want to
        error_moves = errorMoves(move)
        for error_move in error_moves:
            x,y = coordinateAfterMove(position, error_move, legalMoves)
            utility_move += ((1-api.directionProb)/len(error_moves)) * rewardMap[x][y]
        
        if len(maximisingMoves) == 0 or utility_move > maximisingMoves[0][1]:
            maximisingMoves = [(move, utility_move)]
        elif utility_move == maximisingMoves[0][1]:
            maximisingMoves.append((move, utility_move))
    
    return maximisingMoves

def maximisingAction(position, legalMoves, map):
    """
    returns the move with the highest expected utility 
    """
    return random.choice(calculateBestActions(position, legalMoves, map))[0]

def maximumExpectedUtility(position, legalMoves, map):
    """
    returns the expected utility for the the best move 
    """
    return calculateBestActions(position, legalMoves, map)[0][1]

def errorMoves(direction):
    """
    returns a list containing the possible directions pacman might pick due to non-deterministic movement
    """
    return [Directions.LEFT[direction], Directions.RIGHT[direction]]
        
def coordinateAfterMove(position, direction, legalMoves):
    """
    returns pacman's coordinate after move
    """
    x,y = position

    if direction not in legalMoves:
        return position
    elif direction == Directions.NORTH:
        return (x,y+1)
    elif direction == Directions.EAST:
        return (x+1, y)
    elif direction == Directions.SOUTH:
        return (x, y-1)
    elif direction == Directions.WEST:
        return (x-1, y)
    elif direction == Directions.STOP:
        return position
    else:
        raise ValueError("direction argument is not a valid Direction")

def printMap(map):
    """
    print to terminal a representation of pacman's internal map
    """

    max_x = len(map)
    max_y = len(map[0])

    for y in range(max_y-1, -1, -1):
        row = ""
        for x in range(max_x):
            row += (" " + str(int(map[x][y])) + " ") if map[x][y] != "W" else "  W  "
        print(row)