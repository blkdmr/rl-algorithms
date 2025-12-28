# Monte Carlo Tree Search (MCTS) for CartPole — Theory-first Notes

This project applies **Monte Carlo Tree Search (MCTS)** as an *online planning* method: at every timestep, it builds a search tree rooted at the current observation and uses simulated rollouts to pick the next action. 
The implementation uses the classic four MCTS phases—Selection, Expansion, Simulation, Backpropagation—plus a practical step to keep the simulator consistent with the selected tree node. 

## Core idea: planning with sampled futures

MCTS approximates “which action is best now?” by repeatedly sampling possible futures rather than exhaustively enumerating all trajectories. 
Each iteration adds information to a partial tree, concentrating computation on the most promising action sequences while still exploring alternatives. 

## Tree structure and statistics

Each node corresponds to a state (here: a CartPole observation) reached by executing an action sequence from the root. 
Nodes store at least two statistics: a visit count \(N(s)\) and a value estimate \(Q(s)\) (often maintained as total return and/or mean return). 
Over many iterations, \(Q\) becomes a Monte Carlo estimate of expected return under the simulation policy used in rollouts. 

## The four phases of MCTS

### Selection (tree policy)

Selection traverses the current tree from the root to a leaf by repeatedly choosing a “best” child according to a rule that balances: 
- Exploitation: prefer actions with higher estimated value. 
- Exploration: prefer actions that have been tried fewer times to reduce uncertainty. 

A standard choice is UCB1/UCT-style selection, which assigns each child an optimistic score: 

- Mean-value term: \(\bar{Q}(s,a)\) 
- Exploration bonus: \(C \sqrt{\frac{\ln N(s)}{N(s,a)}}\) 

Here, \(C\) is an exploration constant; higher \(C\) explores more, lower \(C\) exploits more. 

### Expansion

When selection reaches a node that is not fully expanded (i.e., there exists at least one action never tried from that node), expansion adds a new child by taking one untried action in the simulator. 
This is what grows the tree over time and creates new frontier states for evaluation. 

### Simulation (rollout policy)

From the newly expanded node (or from a leaf), MCTS runs a rollout using a fast “default policy” (often random actions) to estimate the outcome of that choice. 
The rollout returns a scalar return \(G\) (e.g., sum of rewards until termination or a cutoff), which acts as a noisy sample of the value of the leaf. 
Cutoffs (like maximum rollout reward/steps) reduce computation but change the value target to a truncated-horizon objective. 

### Backpropagation

The rollout return \(G\) is propagated back along the visited path to the root, updating: 
- Visit counts: \(N \leftarrow N + 1\) 
- Value accumulators: \(W \leftarrow W + G\), hence mean \(\bar{Q} = W/N\) 

This makes earlier decisions increasingly informed by deeper sampled outcomes. 

## Choosing the action at the root

After many iterations, MCTS must pick a real action to execute in the real environment. 
A common robust rule is to choose the root child with the highest visit count (most explored), rather than the highest mean value, because visits tend to be more stable under exploration noise. 

## Practical requirement: consistent simulator state

MCTS assumes that when the algorithm is “at” a node, the simulator is in exactly that node’s state before applying additional actions. 
In many Gym-style environments, there is no official “clone/restore full simulator state” API, so implementations often reconstruct node states by resetting and replaying the action sequence from the root.
This approach is pragmatic but environment-dependent: it works best when the environment is deterministic (or seeded) and state reconstruction is faithful. 

## How this code fits the theory (high level)

- It builds a small MCTS tree from the current observation, repeating Selection → Expansion → Rollout → Backprop for a fixed number of iterations. 
- It uses a UCB-style rule to choose children during Selection and a random rollout policy during Simulation. 
- It repeatedly re-synchronizes the simulator to the selected node (via reset + state-setting + replay) to ensure expansions/rollouts start from the correct state. 
- It returns the most visited root action, then repeats the whole process at the next real timestep (online replanning). 

## Key limitations to keep in mind (theory-facing)

- **Value bias from truncation:** any rollout cutoff changes what return is being estimated and can bias action choice toward short-horizon proxies. 
- **Mismatch between rollout policy and optimal policy:** random rollouts give unbiased but high-variance estimates; stronger rollout policies reduce variance but can introduce their own bias if not handled carefully. 
- **Environment stochasticity:** if transitions are stochastic and not controlled (seeding/cloning), replay-based reconstruction can diverge from the intended node state, weakening the meaning of the tree statistics. 
- **Continuous state spaces:** MCTS nodes represent exact visited states; in continuous domains, revisiting the exact same state is unlikely, so planning depth and simulation policy become crucial. 
