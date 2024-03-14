SKIP_GAME = True
if not SKIP_GAME:
    %run dice_game.py
from dice_game import DiceGame
import numpy as np
import itertools as it 

# setting a seed for the random number generator gives repeatable results, making testing easier!
np.random.seed(111)

game = DiceGame()
game.get_dice_state()
reward, new_state, game_over = game.roll((0,))
print(reward)
print(new_state)
print(game_over)
print(game.score)
game.reset()
game = DiceGame()
print(f"The first 5 of {len(game.states)} possible dice rolls are: {game.states[0:5]}")
print(f"The possible actions on any given turn are: {game.actions}")
game = DiceGame()
states, game_over, reward, probabilities = game.get_next_states((0,), (2, 3, 4))
for state, probability in zip(states, probabilities):
    print(f"Would get roll of {state} with probability {probability}")
states, game_over, reward, probabilities = game.get_next_states((0, 1, 2), (2, 2, 5))
print(states)
print(game_over)
print(reward)
from abc import ABC, abstractmethod
from dice_game import DiceGame
import numpy as np


class DiceGameAgent(ABC):
    def __init__(self, game):
        self.game = game
    
    @abstractmethod
    def play(self, state):
        pass


class AlwaysHoldAgent(DiceGameAgent):
    def play(self, state):
        return (0, 1, 2)


class PerfectionistAgent(DiceGameAgent):
    def play(self, state):
        if state == (1, 1, 1) or state == (1, 1, 6):
            return (0, 1, 2)
        else:
            return ()
        
        
def play_game_with_agent(agent, game, verbose=False):
    state = game.reset()
    
    if(verbose): print(f"Testing agent: \n\t{type(agent).__name__}")
    if(verbose): print(f"Starting dice: \n\t{state}\n")
    
    game_over = False
    actions = 0
    while not game_over:
        action = agent.play(state)
        actions += 1
        
        if(verbose): print(f"Action {actions}: \t{action}")
        _, state, game_over = game.roll(action)
        if(verbose and not game_over): print(f"Dice: \t\t{state}")

    if(verbose): print(f"\nFinal dice: {state}, score: {game.score}")
        
    return game.score


def main():
    # random seed makes the results deterministic
    # change the number to see different results
    # or delete the line to make it change each time it is run
    np.random.seed(1)
    
    game = DiceGame()
    
    agent1 = AlwaysHoldAgent(game)
    play_game_with_agent(agent1, game, verbose=True)
    
    print("\n")
    
    agent2 = PerfectionistAgent(game)
    play_game_with_agent(agent2, game, verbose=True)
    

if __name__ == "__main__":
    main()
Testing agent: 
	AlwaysHoldAgent
Starting dice: 
	(1, 1, 2)

Action 1: 	(0, 1, 2)

Final dice: (2, 6, 6), score: 14


Testing agent: 
	PerfectionistAgent
Starting dice: 
	(2, 3, 3)

Action 1: 	()
Dice: 		(3, 4, 5)
Action 2: 	()
Dice: 		(1, 2, 6)
Action 3: 	()
Dice: 		(3, 4, 5)
Action 4: 	()
Dice: 		(1, 2, 5)
Action 5: 	()
Dice: 		(2, 5, 6)
Action 6: 	()
Dice: 		(1, 6, 6)
Action 7: 	()
Dice: 		(1, 2, 6)
Action 8: 	()
Dice: 		(1, 3, 6)
Action 9: 	()
Dice: 		(2, 4, 5)
Action 10: 	()
Dice: 		(1, 5, 6)
Action 11: 	()
Dice: 		(5, 5, 6)
Action 12: 	()
Dice: 		(1, 2, 5)
Action 13: 	()
Dice: 		(2, 3, 6)
Action 14: 	()
Dice: 		(1, 1, 2)
Action 15: 	()
Dice: 		(2, 2, 5)
Action 16: 	()
Dice: 		(1, 3, 4)
Action 17: 	()
Dice: 		(1, 4, 5)
Action 18: 	()
Dice: 		(1, 3, 5)
Action 19: 	()
Dice: 		(1, 3, 4)
Action 20: 	()
Dice: 		(4, 4, 6)
Action 21: 	()
Dice: 		(1, 4, 6)
Action 22: 	()
Dice: 		(1, 3, 5)
Action 23: 	()
Dice: 		(1, 3, 6)
Action 24: 	()
Dice: 		(5, 5, 6)
Action 25: 	()
Dice: 		(3, 4, 5)
Action 26: 	()
Dice: 		(2, 3, 6)
Action 27: 	()
Dice: 		(4, 4, 6)
Action 28: 	()
Dice: 		(1, 3, 6)
Action 29: 	()
Dice: 		(2, 3, 4)
Action 30: 	()
Dice: 		(1, 4, 6)
Action 31: 	()
Dice: 		(2, 4, 4)
Action 32: 	()
Dice: 		(3, 6, 6)
Action 33: 	()
Dice: 		(1, 4, 6)
Action 34: 	()
Dice: 		(2, 5, 6)
Action 35: 	()
Dice: 		(1, 5, 6)
Action 36: 	()
Dice: 		(1, 5, 5)
Action 37: 	()
Dice: 		(1, 5, 6)
Action 38: 	()
Dice: 		(1, 1, 1)
Action 39: 	(0, 1, 2)

Final dice: (6, 6, 6), score: -20
import numpy as np
import itertools as it
class MyAgent(DiceGameAgent):
    def __init__(self, game):
        """
        if your code does any pre-processing on the game, you can do it here
        you can always access the game with self.game
        """
        # this calls the superclass constructor (does self.game = game)
        super().__init__(game) 
        
        
    def play(self, state):
        """
        given a state, return the chosen action for this state
        at minimum you must support the basic rules: three six-sided fair dice
        if you want to support more rules, use the values inside self.game, e.g.
            the input state will be one of self.game.states
            you must return one of self.game.actions
        read the code in dicegame.py to learn more
        """
        list_of_numbers = [1,2,3,4,5,6]
        combinations_list = []
        for n in range(len(list_of_numbers) + 1) :
            combinations_list += list(it.combinations_with_replacement(list_of_numbers, 3))
        normal_combinations_set = set(combinations_list)
        normal_combinations_list = [x for x in normal_combinations_set]
        normal_combinations_list.sort()
        sums_of_normal_combinations = [sum(x) for x in normal_combinations_list]
        set_of_normal_sums = list(set(sums_of_normal_combinations))
        normal_combinations_and_sums = [[(x), sum(x)] for x in normal_combinations_list]
        normal_sums_and_probabilities = [(x,round((sums_of_normal_combinations.count(x) / 56)*100, 2)) for x in set_of_normal_sums]
        real_combinations_list = []
        for x in normal_combinations_list : 
            copy_member = list(x)
            for y in copy_member : 
                if x.count(y) > 1 : 
                    copy_member[copy_member.index(y)] = 7-y 
            real_combinations_list.append(copy_member)
        real_and_normal_combinations = []
        for x,y in enumerate(normal_combinations_list) : 
            temporary_list = []
            temporary_list.append(normal_combinations_list[x])
            temporary_list.append(real_combinations_list[x])
            real_and_normal_combinations.append(temporary_list) 
        for x in real_and_normal_combinations : 
            x[0] = list(x[0])
            x.append(sum(x[1]))
        for x in real_and_normal_combinations : 
            for y in normal_sums_and_probabilities : 
                if x[2] == y[0] : 
                    x.append(y[1]) 
        sums_15_to_18 = [x[0] for x in real_and_normal_combinations if x[2] >= 15]
        sums_11_to_14 = [x[0] for x in real_and_normal_combinations if x[2] in [11,12,13,14]]
        sums_7_to_10 = [x[0] for x in real_and_normal_combinations if x[2] in [7,8,9,10]]
        sums_3_to_6 = [x[0] for x in real_and_normal_combinations if x[2] in [3,4,5,6]]
        permutacije_1 = [list(it.permutations(x)) for x in [[1,1,1],[1,1,5],[1,1,6],[2,2,2],[2,2,5],[2,2,6],[4,5,6],[1,1,4]]]
        permutations_list_1 = []
        for x in permutacije_1 : 
            for y in x : 
                permutations_list_1.append(list(y))
        permutacije_2 = [list(it.permutations(x)) for x in [[1,1,3]]]
        permutations_list_2 = []
        for x in permutacije_2 : 
            for y in x : 
                permutations_list_2.append(list(y))
        permutacije_3 = [list(it.permutations(x)) for x in [[1,1,2],[2,2,3],[2,2,4],[3,3,3],[3,3,4]]]
        permutations_list_3 = []
        for x in permutacije_3 : 
            for y in x : 
                permutations_list_3.append(list(y))
        permutacije_4 = [list(it.permutations(x)) for x in [[1,2,2]]]
        permutations_list_4 = []
        for x in permutacije_4 : 
            for y in x : 
                permutations_list_4.append(list(y))
        permutacije_5 = [list(it.permutations(x)) for x in [[1,4,6]]]
        permutations_list_5 = []
        for x in permutacije_5 : 
            for y in x : 
                permutations_list_5.append(list(y))
        permutacije_6 = [list(it.permutations(x)) for x in [[2,3,6]]]
        permutations_list_6 = []
        for x in permutacije_6 : 
            for y in x : 
                permutations_list_6.append(list(y))
        permutacije_7 = [list(it.permutations(x)) for x in [[1,5,6], [2,4,5], [2,4,6], [2,5,6], [3,3,5], [3,3,6], [3,4,5],
                                                            [3,4,6], [3,5,6], [4,4,5], [4,4,6]]]
        permutations_list_7 = []
        for x in permutacije_7 : 
            for y in x : 
                permutations_list_7.append(list(y))
        permutacije_8 = [list(it.permutations(x)) for x in [[1,2,4], [1,2,5], [1,2,6]]]
        permutations_list_8 = []
        for x in permutacije_8 : 
            for y in x : 
                permutations_list_8.append(list(y))
        permutacije_9 = [list(it.permutations(x)) for x in [[1, 3, 3], [1, 3, 4], [1, 3, 5], [1, 3, 6], [1, 4, 4], [1, 4, 5], 
                                                            [2, 3, 3], [2, 3, 4], [2, 3, 5], [2, 4, 4]]]
        permutations_list_9 = []
        for x in permutacije_9 : 
            for y in x : 
                permutations_list_9.append(list(y))
        permutacije_10 = [list(it.permutations(x)) for x in [[3, 4, 4], [3, 5, 5], [4, 4, 4], [4, 5, 5], [5, 5, 6], [5, 6, 6]]]
        permutations_list_10 = []
        for x in permutacije_10 : 
            for y in x : 
                permutations_list_10.append(list(y))
        permutacije_11 = [list(it.permutations(x)) for x in [[1,2,3]]]
        permutations_list_11 = []
        for x in permutacije_11 : 
            for y in x : 
                permutations_list_11.append(list(y))
        permutacije_12 = [list(it.permutations(x)) for x in [[1, 5, 5], [1, 6, 6], [2, 5, 5], [2, 6, 6], [3, 6, 6], [4, 6, 6]]]
        permutations_list_12 = []
        for x in permutacije_12 : 
            for y in x : 
                permutations_list_12.append(list(y))
        if list(state) in permutations_list_1 : 
            return (0,1,2) 
        elif list(state) in permutations_list_2 : 
            indices = []
            for x,y in enumerate(list(state)) : 
                if y == 1 : 
                    indices.append(x)
            return tuple(indices)
        elif list(state) in permutations_list_3 : 
            if list(state) == [3,3,3] : 
                return (0,1,2)
            indices = []
            for x,y in enumerate(list(state)) : 
                if list(state).count(y) == 2 : 
                    indices.append(x)
            return tuple(indices)
        elif list(state) in permutations_list_4 : 
            indices = []
            for x,y in enumerate(list(state)) : 
                if y == 2 : 
                    indices.append(x)
            return tuple(indices)
        elif list(state) in permutations_list_5 : 
            indices = []
            for x,y in enumerate(list(state)) : 
                if y == 4 or y == 6 : 
                    indices.append(x)
            return tuple(indices)
        elif list(state) in permutations_list_6 : 
            indices = []
            for x,y in enumerate(list(state)) : 
                if y == 3 or y == 6 : 
                    indices.append(x)
            return tuple(indices)
        elif list(state) in permutations_list_7 : 
            return (0,1,2)
        elif list(state) in permutations_list_8 : 
            indices = []
            for x,y in enumerate(list(state)) : 
                if y == 3 or y == 6 : 
                    indices.append(x)
            return tuple(indices)
        elif list(state) in permutations_list_9 : 
            indices = []
            for x,y in enumerate(list(state)) : 
                if y not in [1,2] :
                    indices.append(x)
            return tuple(indices)
        elif list(state) in permutations_list_10 : 
            return ()
        elif list(state) in permutations_list_11 : 
            indices = []
            for x,y in enumerate(list(state)) : 
                if y == 1 or y == 2 : 
                    indices.append(x)
            return tuple(indices)
        elif list(state) in permutations_list_12 : 
            indices = []
            for x,y in enumerate(list(state)) : 
                if y in [1,2,3,4] : 
                    indices.append(x)
            return tuple(indices)
SKIP_TESTS = True

def tests():
    import time

    total_score = 0
    total_time = 0
    n = 10

    np.random.seed()

    print("Testing basic rules.")
    print()

    game = DiceGame()

    start_time = time.process_time()
    test_agent = MyAgent(game)
    total_time += time.process_time() - start_time

    for i in range(n):
        start_time = time.process_time()
        score = play_game_with_agent(test_agent, game)
        total_time += time.process_time() - start_time

        print(f"Game {i} score: {score}")
        total_score += score

    print()
    print(f"Average score: {total_score/n}")
    print(f"Total time: {total_time:.4f} seconds")
    
if not SKIP_TESTS:
    tests()
TEST_EXTENDED_RULES = False 
import time 
def extended_tests():
    total_score = 0
    total_time = 0
    n = 10

    print("Testing extended rules – two three-sided dice.")
    print()

    game = DiceGame(dice=2, sides=3)

    start_time = time.process_time()
    test_agent = MyAgent(game)
    total_time += time.process_time() - start_time

    for i in range(n):
        start_time = time.process_time()
        score = play_game_with_agent(test_agent, game)
        total_time += time.process_time() - start_time

        print(f"Game {i} score: {score}")
        total_score += score

    print()
    print(f"Average score: {total_score/n}")
    print(f"Average time: {total_time/n:.5f} seconds")

if not SKIP_TESTS and TEST_EXTENDED_RULES:
    extended_tests()