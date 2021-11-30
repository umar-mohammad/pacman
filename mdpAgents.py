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
        self.utility_map = None
        self.reward_values = None


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
                "ghost" : -300,
                "edible_ghost" : 100,
                "food" : 10,
                "pacman" : -20,
                "capsule" : 10.5,
                "deathzone" : -300,
                "wall" : "W"
        }
        
        map = []
        width = 0
        height = 0
        
        for x,y in api.corners(state):
            width = max(x, width)
            height = max(y, height)
        
        if width == 19 and height == 10:
            # mediumClassic map
            self.reward_values = MEDIUM_CLASSIC_REWARDS
        else:
            self.reward_values = SMALL_GRID_REWARDS
        
        for _ in range(width+1):
            map.append([self.reward_values["empty"]] * (height+1))
            
        self.utility_map = map
        
        

    def final(self, state):
        self.utility_map = None
        self.reward_values = None


    def getAction(self, state):
        legal_moves = api.legalActions(state)
        value_function = self.value_iteration(self.get_reward_map(state), gamma=0.9, epsilon=1)
        # print_map(value_function)
        max_move = get_optimal_action(api.whereAmI(state), legal_moves, value_function)
        return api.makeMove(max_move, legal_moves)


    def value_iteration(self, utility_map, gamma=0.9, epsilon=5):
        """
        returns the utility values for each coordinate of the map
        """
        map = self.utility_map # copies the utility map from the previous time step, which allows the algorithm to converge faster
        
        # do value iteration until cumulative change in value is less than epsilon
        while True:
            old_map = copy_map(map)
            delta = 0
            
            for x in range(len(map)):
                for y in range(len(map[0])):
                    utility = utility_map[x][y]
                    if utility == "W":
                        map[x][y] = "W"
                    else:
                        updated_value = utility_map[x][y] + gamma * maximum_expected_utility((x,y), get_legal_actions((x,y), old_map), old_map)
                        delta += abs(updated_value - old_map[x][y])
                        map[x][y] = updated_value
            if delta <= epsilon: 
                break
            
        self.utility_map = map
        return map
    
    
    def get_reward_map(self, state):
        """
        return a 2D list representing the game state with reward values
        """
        # build initial map with empty spaces for every position in grid
        reward_map = self.create_empty_map(state)
        
        # update rewards
        self.add_wall_reward(state, reward_map)
        self.add_capsule_reward(state, reward_map)
        self.add_food_reward(state, reward_map)
        self.add_pacman_reward(state, reward_map)
        self.add_ghosts_reward(state, reward_map)
        self.add_spawn_area_reward(state, reward_map)
        
        return reward_map


    def create_empty_map(self, state):
        """
        returns an empty reward map with the same dimensions as the game map
        """
        map = []
        width = 0
        height = 0
        
        for x,y in api.corners(state):
            width = max(x, width)
            height = max(y, height)
            
        for _ in range(width+1):
            map.append([self.reward_values["empty"]] * (height+1))
        
        return map
    
    
    def add_pacman_reward(self, state, map):
        """
        add pacman rewards to map
        """
        pacman_x, pacman_y = api.whereAmI(state)
        map[pacman_x][pacman_y] += self.reward_values["pacman"]
    
    
    def add_wall_reward(self, state, map):
        """
        add wall rewards to map
        """
        walls = api.walls(state)
        for x, y in walls: map[x][y] = self.reward_values["wall"]
    
    
    def add_capsule_reward(self, state, map):
        """
        add capsule rewards to map
        """
        capsules = api.capsules(state)
        for x,y in capsules: map[x][y] += self.reward_values["capsule"]
    
    
    def add_food_reward(self, state, map):
        """
        add food rewards to map
        """
        food = api.food(state)
        for x,y in food: map[x][y] += self.reward_values["food"]
    
    
    def add_ghosts_reward(self, state, map):
        """
        add ghost rewards to map
        """
        MAX_GHOST_EDIBLE_TIME = 40 
        ghosts = api.ghosts(state)
        edible_time = dict(api.ghostStatesWithTimes(state))
        for pos in ghosts:
            x,y = util.nearestPoint(pos)
            value = -10 + ((self.reward_values["edible_ghost"] * edible_time[pos])/MAX_GHOST_EDIBLE_TIME) if pos in edible_time and edible_time[pos] > 0 else self.reward_values["ghost"]
            map[x][y] += value

            for position, steps in get_surrounding_positions(pos, 3, map):
                x,y = position
                map[x][y] += (value / steps)
            
            for position, steps in get_line_of_sight_positions(pos, 5, map):
                x,y = position
                map[x][y] += (value / steps)
    
    
    def add_spawn_area_reward(self, _state, map):
        """
        add spawn area rewards to map
        """
        # check which map pacman is playing
        map_width, map_height = len(map), len(map[0])
        if map_width == 20 and map_height == 11:
            # mediumClassic map
            # make ghost spawn zone an undersirable area
            for x in range(8,12,1):
                map[x][5] += self.reward_values["deathzone"]

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


def get_surrounding_positions(position, steps, map):  
    """
    returns the legal positions that are within x steps of the ghost
    """
    x,y = util.nearestPoint(position)
    surrounding_positions = []
    surrounding_positions.extend(surrounding_positions_helper((x,y), Directions.NORTH, 0, steps, map))
    surrounding_positions.extend(surrounding_positions_helper((x,y), Directions.EAST, 0, steps, map))
    surrounding_positions.extend(surrounding_positions_helper((x,y), Directions.SOUTH, 0, steps, map))
    surrounding_positions.extend(surrounding_positions_helper((x,y), Directions.WEST, 0, steps, map))

    return surrounding_positions


def surrounding_positions_helper(position, direction, steps_taken, steps, map):
    """
    get_surrounding_positions() helper function, explores all coordinates in one main direction to check if they are legal positions 
    and recrusively try to explore perpendicular directions to the main
    """
    wall = "W"
    x, y = util.nearestPoint(position)
    surrounding_positions = []
    x_add, y_add  = Actions._directions[direction]
    
    for i in range(1, (steps-steps_taken)+1, 1):
        if map[x+(i*x_add)][y+(i*y_add)] != wall:
            surrounding_positions.append(((x+x_add, y+y_add), steps_taken+i))
            surrounding_positions.extend(surrounding_positions_helper((x+(i*x_add), y+(i*y_add)), Directions.LEFT[direction], steps_taken+i, steps, map))
            surrounding_positions.extend(surrounding_positions_helper((x+(i*x_add), y+(i*y_add)), Directions.RIGHT[direction], steps_taken+i, steps, map))
        else:
            break
    
    return surrounding_positions


def get_line_of_sight_positions(position, steps, map):
    """
    returns the legal positions that are on the same latitude and longitude 
    """
    x,y = util.nearestPoint(position)
    los_positions = []
    los_positions.extend(line_of_sight_helper((x,y), Directions.NORTH, steps, map))
    los_positions.extend(line_of_sight_helper((x,y), Directions.EAST, steps, map))
    los_positions.extend(line_of_sight_helper((x,y), Directions.SOUTH, steps, map))
    los_positions.extend(line_of_sight_helper((x,y), Directions.WEST, steps, map))
    
    return los_positions
    
    
def line_of_sight_helper(position, direction, step, map):
    """
    explores on direction until it meets a wall or reaches the step limit and returns all the legal positions
    """
    wall = "W"
    x, y = util.nearestPoint(position)
    los_positions = []
    x_add, y_add  = Actions._directions[direction]
    current_step = 1
    
    while(current_step <= step and map[x+(current_step*x_add)][y+(current_step*y_add)] != wall):
        los_positions.append(((x+(current_step*x_add), y+(current_step*y_add)), current_step))
        current_step+=1
        
    return los_positions


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

def maximum_expected_utility(position, legal_moves, reward_map):
    """
    calculates the maximum expected utility for a certain position on the map
    """
    best_move_utility = 0

    for move in legal_moves:
        x,y = get_coordinate_after_move(position, move, legal_moves)
        utility_move = api.directionProb * reward_map[x][y]

        # calculate the rewards if the move does not execute like we want to
        error_moves = get_error_moves(move)
        for error_move in error_moves:
            x,y = get_coordinate_after_move(position, error_move, legal_moves)
            utility_move += ((1-api.directionProb)/len(error_moves)) * reward_map[x][y]
        
        best_move_utility = max(best_move_utility, utility_move)
    
    return best_move_utility

        
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
    returns move with the highest expected utility 
    """
    return random.choice(calculate_best_actions(position, legal_moves, map))[0]


def get_error_moves(direction):
    """
    returns a list containing the possible directions pacman might pick due to non-deterministic movement
    """
    return [Directions.LEFT[direction], Directions.RIGHT[direction]]
   
     
def get_coordinate_after_move(position, direction, legal_moves):
    """
    returns pacman's coordinate after moving in a certain direction
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
            row += (" " + str(int(map[x][y])) + " ") if map[x][y] != "W" else " W "
        print(row)