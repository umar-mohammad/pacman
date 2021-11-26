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

from game import Directions, Actions

class MDPAgent(game.Agent):


    def __init__(self):
        self.reward_map = None
        self.map_values = None
        self.ghost_previous = None


    def registerInitialState(self, state):
        """
        build initial map to be used for value iteration 
        """
        SMALL_GRID_REWARDS = {
            "empty" : 0.5,
            "ghost" : -6,
            "food" : 3,
            "pacman" : 0,
            "wall" : "W"
        }
        MEDIUM_CLASSIC_REWARDS = {
                "empty" : -1,
                "ghost" : -500,
                "edible_ghost" : 100,
                "food" : 10,
                "pacman" : -5,
                "capsule" : 10.5,
                "deathzone" : -25,
                "wall" : "W"
        }

        reward_map = []
        width=0
        height=0

        for x,y in api.corners(state):
            width = max(x, width)
            height = max(y, height)
        
        if width == 19 and height == 10:
            # mediumClassic map
            self.map_values = MEDIUM_CLASSIC_REWARDS
        else:
            self.map_values = SMALL_GRID_REWARDS

        for _ in range(width+1):
            reward_map.append([0] * (height+1))
        self.reward_map = reward_map
        self.ghost_previous = api.ghosts(state)
        
        

    def final(self, state):
        self.reward_map = None
        self.map_values = None
        self.ghost_previous = None


    def getAction(self, state):
        legal_moves = api.legalActions(state)
        max_move = get_optimal_action(api.whereAmI(state), legal_moves, self.value_iteration(self.get_reward_map(state), 0.999))
        return api.makeMove(max_move, legal_moves)


    def value_iteration(self, rewards, gamma=0.9):
        """
        returns the utility values for each coordinate of the map
        """
        map = self.reward_map # copies the reward map from the previous time step, which allows the algorithm to converge faster
        
        while True:
            old_map = copy_map(map)
            delta = 0
            for x in range(len(map)):
                for y in range(len(map[0])):
                    reward = rewards[x][y]
                    if reward == "W":
                        map[x][y] = "W"
                    else:
                        updated_value = rewards[x][y] + gamma * weighted_expected_utility((x,y), get_legal_actions((x,y), old_map), old_map, 0.9, 0.1)
                        delta += abs(updated_value - old_map[x][y])
                        map[x][y] = updated_value
            if delta <= 5: 
                break
            
        self.reward_map = map
        return map
    
    
    def get_reward_map(self, state):
        """
        return a 2D list representing the game state with reward values
        """
        map = []

        pacman_x, pacman_y = api.whereAmI(state)
        walls = api.walls(state)
        corners = api.corners(state)
        ghosts = api.ghosts(state)
        edible_ghost = dict(api.ghostStatesWithTimes(state))
        food = api.food(state)
        capsules = api.capsules(state)

        # build initial map with empty spaces for every position in grid
        width = 0
        height = 0
        for x,y in corners:
            width = max(x, width)
            height = max(y, height)
        for _ in range(width+1):
            map.append([self.map_values["empty"]] * (height+1))

        # place all game elements to the map
        map[pacman_x][pacman_y] = self.map_values["pacman"]
        for x,y in capsules: map[x][y] += self.map_values["capsule"]
        for x,y in food: map[x][y] = self.map_values["food"]
        for x, y in walls: map[x][y] = self.map_values["wall"]
        
        for i, pos in enumerate(ghosts):
            x,y = util.nearestPoint(pos)
            value = ((self.map_values["edible_ghost"] * edible_ghost[pos])/40) if pos in edible_ghost and edible_ghost[pos] > 0 else self.map_values["ghost"]
            map[x][y] += value

            for future_position, steps in get_ghost_future_position(pos, get_ghost_direction(self.ghost_previous[i], pos, map), 4, map):
                x,y = future_position
                map[x][y] += value * (1/steps)
            
        if width == 19 and height == 10:
            # mediumClassic map
                for x in range(8,12,1):
                    map[x][5] += self.map_values["deathzone"]
        
        self.ghost_previous = ghosts
        return map
    
    
def get_ghost_direction(previous, now, map):
    """
    returns the direction the ghost is headed in based on its previous and current positions
    returns None if we cannot predict the direction the ghost is headed in
    """
    direction_vector = (now[0] - previous[0], now[1] - previous[1])
    if abs(direction_vector[0]) > 1 or abs(direction_vector[1]) > 1:
        # ghost got eaten and spawned again 
        # we cannot predict wehre it will go next
        return None
    
    if len(get_legal_actions(util.nearestPoint(now), map)) >= 4:
        # ghost is at a junction and it will pick a random direction 
        # hence we cannot predict where it will go
        return None
    
    if direction_vector[0] == 0:
        if direction_vector[1] > 0: return Directions.NORTH
        else: return Directions.SOUTH
    else:
        if direction_vector[0] > 0: return Directions.EAST
        else: return Directions.WEST


def get_legal_actions(position, map):
    """
    returns all legal moves for a particular position on the map
    """
    moves=[(0,1), (1, 0), (0, -1), (-1, 0)]
    legal_actions=[Directions.STOP]
    wall = "W"
    for i, move in enumerate(moves):
        x,y = (move[0] + position[0], move[1] + position[1])
        if map[x][y] != wall:
            if i == 0: legal_actions.append(Directions.NORTH)
            elif i == 1: legal_actions.append(Directions.EAST)
            elif i == 2: legal_actions.append(Directions.SOUTH)
            elif i == 3: legal_actions.append(Directions.WEST) 
    return legal_actions


def get_ghost_future_position(position, direction, steps, map):
    """
    returns the positions where the ghost can be in after a number of steps
    """
    x,y = util.nearestPoint(position)
    future_positions = []
    back = None if direction == None else Directions.REVERSE[direction]
    future_positions.extend(ghost_future_helper((x,y), Directions.NORTH, 0, int(steps/2) if back == Directions.NORTH else steps, map))
    future_positions.extend(ghost_future_helper((x,y), Directions.EAST, 0, int(steps/2) if back == Directions.EAST else steps, map))
    future_positions.extend(ghost_future_helper((x,y), Directions.SOUTH, 0, int(steps/2) if back == Directions.SOUTH else steps, map))
    future_positions.extend(ghost_future_helper((x,y), Directions.WEST, 0, int(steps/2) if back == Directions.WEST else steps, map))

    return future_positions


def ghost_future_helper(position, direction, steps_taken, steps, map):
    """
    helper function that helps predict future ghost positions
    """
    wall = "W"
    x, y = util.nearestPoint(position)
    future_positions = []
    x_add, y_add  = Actions._directions[direction]
    for i in range(1, (steps-steps_taken)+1, 1):
        if map[x+(i*x_add)][y+(i*y_add)] != wall:
            future_positions.append(((x+x_add, y+y_add), steps_taken+i))
            future_positions.extend(ghost_future_helper((x+(i*x_add), y+(i*y_add)), Directions.LEFT[direction], steps_taken+i, steps, map))
            future_positions.extend(ghost_future_helper((x+(i*x_add), y+(i*y_add)), Directions.RIGHT[direction], steps_taken+i, steps, map))
        else:
            break
    
    return future_positions


def copy_map(map):
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


def weighted_expected_utility(position, legal_moves, reward_map, positive_weight=0.85, negative_weight=0.15):
    """
    calculates a weighted expected utility for a certain position on the map, taking into account both
    the maximum and minimum reward you can get for a cell, this helps adjust how optimistic pacman is
    """
    best_move_utility = 0
    worst_move_utility = 0

    for move in legal_moves:
        x,y = get_coordinate_after_move(position, move, legal_moves)
        utility_move = api.directionProb * reward_map[x][y]

        # calculate the rewards if the move does not execute like we want to
        error_moves = get_error_moves(move)
        for error_move in error_moves:
            x,y = get_coordinate_after_move(position, error_move, legal_moves)
            utility_move += ((1-api.directionProb)/len(error_moves)) * reward_map[x][y]
        
        best_move_utility = max(best_move_utility, utility_move)
        worst_move_utility = min(worst_move_utility, utility_move)
    
    return positive_weight * best_move_utility + negative_weight * worst_move_utility

        
def calculate_best_actions(position, legal_moves, utility_map):
    """
    calculates the best actions based on the utility values of pacman
    """
    # maximisingMoves is a list containing all move-utility tuples that have the highest utility value
    # this allows us to choose a random move which makes pacman's movement non-deterministic 
    # when multiple moves have equal utility value which helps it not get stuck in corners
    maximising_moves = [] 

    for move in legal_moves:
        x,y = get_coordinate_after_move(position, move, legal_moves)
        utility_move = api.directionProb * utility_map[x][y]

        # calculate the rewards if the move does not execute like we want to
        error_moves = get_error_moves(move)
        for error_move in error_moves:
            x,y = get_coordinate_after_move(position, error_move, legal_moves)
            utility_move += ((1-api.directionProb)/len(error_moves)) * utility_map[x][y]
        
        if len(maximising_moves) == 0 or utility_move > maximising_moves[0][1]:
            maximising_moves = [(move, utility_move)]
        elif utility_move == maximising_moves[0][1]:
            maximising_moves.append((move, utility_move))
    
    return maximising_moves


def get_optimal_action(position, legal_moves, map):
    """
    returns the move with the highest expected utility 
    """
    return random.choice(calculate_best_actions(position, legal_moves, map))[0]


def get_error_moves(direction):
    """
    returns a list containing the possible directions pacman might pick due to non-deterministic movement
    """
    return [Directions.LEFT[direction], Directions.RIGHT[direction]]
   
     
def get_coordinate_after_move(position, direction, legal_moves):
    """
    returns pacman's coordinate after move
    """
    x,y = position

    if direction not in legal_moves:
        return position
    else:
        x_offset, y_offset = Actions._directions[direction]
        return (x + x_offset, y + y_offset)


def print_map(map):
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