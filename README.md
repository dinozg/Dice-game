**Title: 3-Dice Game AI Agent**

**Description** 

This project develops a strategic AI agent for a 3-dice game, focusing on maximizing the sum of the dice. 
The agent uses probability calculations and strategic decision-making to determine when to reroll dice.

**Key Features**

* Statistical Analysis:
    * Calculates the probability distribution of dice sum combinations.
    * Understands the Gaussian nature of the outcome probabilities.
* Strategic Decision-Making:
     * Identifies combinations within sum ranges (e.g., 18-15, 14-11) for potential rerolls.
     * Evaluates the probability of improving the sum by rerolling a specific die.
    * Only rerolls a die when the probability of a better outcome exceeds 50%.
* Technical Implementation:
    * Leverages the `itertools` module to handle dice permutations.

**How the Agent Works**

The agent's approach is rooted in probability and strategic optimization:

1. **Analyzing Outcomes:** It understands the likelihood of various sum combinations, prioritizing those in the middle of the Gaussian curve.
2. **Evaluating Rerolls:**  For specific combinations, it calculates the probability of achieving a higher sum by rerolling one die.
3. **Informed Decisions:** Rerolls are only made when the calculated probability of improvement surpasses a 50% threshold.

**Future Development**

* Explore advanced machine learning techniques to enhance the agent's strategy.
* Implement a user-friendly interface for interacting with the agent.
