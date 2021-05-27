# GVizDoom
Platform for training generalizable deep reinforcement learning agents

## Installation
```
git clone https://github.com/TTomilin/GVizDoom
```

## Algorithms implemented
* DQN DRQN Dueling-DQN C51-DQN DFP

## Scenarios and tasks implemented
### Defend the Center
#### Train
* Level 1
  * Default
  * Gore
  * Mossy Bricks
  * Stone Wall
  * Flying Enemies
  * Fast Enemies
  * Fuzzy Enemies
  * Resized Enemies
#### Test
* Level 2
  * Gore + Mossy Bricks
  * Resized Fuzzy Enemies
  * Stone Wall + Flying Enemies
* Level 3
  * Resized Flying Enemies + Mossy Bricks
  * Gore + Stone Wall + Fuzzy Enemies
  * Fast Resized Enemies + Gore
* Level 4
  * Complete
  
#### Health Gathering
#### Train
* Level 1
  * Default
  * Lava 
  * Slime
  * Supreme
  * Poison
  * Obstacles
  * Stimpacks 
  * Shaded Kits
  * Resized_kits
#### Test
* Level 2
  * Slimy Obstacles
  * Shaded Stimpacks    
  * Supreme Poison
* Level 3
  * Lava + Supreme + Short Agent 
  * Obstacles + Slime + Stimpacks
  * Poison + Resized Shaded Kits 
* Level 4
  * Complete
#### Seek and Kill
#### Dodge Projectiles

## Command line usage

### Available input arguments
| Argument                    | Default Value   | Description |
| --------------------------- |:---------------:| ----------- |
| -a --algorithm              | None            | DRL algorithm used to construct the model [DQN, DRQN, Dueling_DQN, DFP, C51] (case-insensitive) | 
| --task-prioritization       | False           | Scale the target weights according to task difficulty calculating the loss | 
| --target-model              | True            | Use a target model for stability in learning | 
| --KPI-update-frequency      | 30              | Number of episodes after which to update the Key Performance Indicator of a task for prioritization | 
| --target-update-frequency   | 3000            | Number of iterations after which to copy the weights of the online network to the target network | 
| --frames_per_action         | 4               | Frame skip count. Number of frames to stack upon each other as input to the model |
| --distributional            | False           | Learn to approximate the distribution of returns instead of the expected return |
| --double-dqn                | False           | Use the online network to predict the actions, and the target network to estimate the Q value |
| --decay-epsilon             | True            | Use epsilon decay for exploration | 
| --train                     | True            | Train or evaluate the agent | 
| -o --observe                | 10000           | Number of iterations to collect experience before training | 
| -e --explore                | 50000           | Number of iterations to decay the epsilon for | 
| --learning-rate             | 0.0001          | Learning rate of the model optimizer | 
| --max-train-iterations      | 10000000        | Maximum iterations of training |
| --initial-epsilon           | 1.0             | Starting value of epsilon, which represents the probability of exploration | 
| --final-epsilon             | 0.001           | Final value of epsilon, which represents the probability of exploration | 
| -g --gamma                  | 0.99            | Value of the discount factor for future rewards | 
| --batch-size                | 32              | Number samples in a single training batch | 
| --multi-step                | 1               | Number of steps to aggregate before bootstrapping |
| --trainer-threads           | 1               | Number of threads used for training the model |
| --load-model                | False           | Load existing weights or train from scratch | 
| --model-save-frequency      | 5000            | Number of iterations after which to save a new version of the model | 
| --model-name-addition       | [Empty String]  | An additional identifier to the name of the model. Used to better differentiate stored data | 
| -n --noisy-nets             | False           | Inject noise to the parameters of the last Dense layers to promote exploration | 
| --prioritized-replay        | False           | Use PER (Prioritized Experience Reply for storing and sampling the transitions | 
| --load-experience           | False           | Load existing experience into the replay buffer | 
| --save-experience           | False           | Store the gathered experience buffer externally | 
| --memory-capacity           | 50000           | Number of most recent transitions to store in the replay buffer | 
| --memory-storage-size       | 5000            | Number of most recent transitions to store externally from the replay buffer if saving is enabled | 
| --memory-update_frequency   | 5000            | Number of iterations after which to externally store the most recent experiences | 
| -s --scenario               | None            | Name of the scenario e.g., `defend_the_center` (case-insensitive) | 
| -t --tasks                  | default         | List of tasks, e.g., `default gore stone_wall` (case-insensitive) | 
| --trained-task              | None            | Name of the trained model. Used for evaluation | 
| --seed                      | None            | Used to fix the game instance to be deterministic | 
| -m --max-epochs             | 10000           | Maximum number of episodes per scenario task | 
| -v --visualize              | False           | Visualize the interaction of the agent with the environment | 
| --render-hud                | True            | Render the in-game hud, which displays health, ammo, armour, etc. | 
| --frame-width               | 84              | Number of pixels to which the width of the frame is scaled down to | 
| --frame-height              | 84              | Number of pixels to which the height of the frame is scaled down to | 
| --append-statistics         | True            | Append the rolling statistics to an existing file or overwrite | 
| --statistics-save-frequency | 5000            | Number of iterations after which to write newly aggregated statistics | 
| --train-report-frequency    | 1000            | Number of iterations after which the training progress is reported |


#### Examples
Train DQN
```
python3 run.py --algorithm dqn --scenario defend_the_center --tasks default gore stone_wall fast_enemies \
    mossy_bricks fuzzy_enemies flying_enemies resized_enemies --seed 1111 --model-name-addition _SEED_1111
```

Train RAINBOW
```
python3 run.py --algorithm dueling_dqn --double-dqn True --scenario defend_the_center --tasks default gore stone_wall \
    fast_enemies mossy_bricks fuzzy_enemies flying_enemies resized_enemies --noisy-nets True \
    --prioritized-replay True --multi-step 3 --distributional True --double-dqn True \
    --task-prioritization True --seed 1111 --model-name-addition _SEED_1111 
```

Test Agent
```
python3 run.py --algorithm dueling_dqn --scenario defend_the_center --train False \
    --tasks gore_mossy_bricks --noisy-nets True --load-model True --observe 0 \
    --decay-epsilon False --max-epochs 100 --trained-model multi_SEED_1111
```
