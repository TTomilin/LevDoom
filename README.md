# LevDoom
LevDoom is a benchmark with difficulty levels based on visual modifications, intended for research in generalization of deep reinforcement learning agents. The benchmark is based upon [ViZDoom](https://github.com/mwydmuch/ViZDoom), a platform addressed to pixel based learning in the FPS game domain.

For more details please refer to our [CoG2022](https://ieee-cog.org/2022/assets/papers/paper_30.pdf) paper.

![Default](assets/gifs/scenarios.gif)

## Installation
1. Install the dependencies for ViZDoom: [Linux](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#-linux), [MacOS](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#-linux) or [Windows](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#-windows).
2. Clone the repository
```bash
$ git clone https://github.com/TTomilin/LevDoom
```
3. Navigate into the repository
```bash
$ cd LevDoom
```
4. Install the dependencies 
```bash 
$ python install setup.py
```
## Environments
The benchmark consists of 4 scenarios, each with multiple environments of increasing difficulty.

### Scenarios
- Defend the Center
- Health Gathering
- Seek and Slay
- Dodge Projectiles

### Modification Types
- Textures
- Obstacles
- Entity Size
- Entity Type
- Entity Rendering
- Enemy Speed
- Agent Height

### Difficulty Levels
The number of combined modifications determines the difficulty level.

| Scenario          | Level 0                                                          | Level 1                                                                   | Level 2                                                                                                | Level 3                                                                                                                    | Level 4                                                            |
|-------------------|------------------------------------------------------------------|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Defend the Center | ![Default](assets/images/defend_the_center/Level0_Default.png?raw=true) | ![Gore](assets/images/defend_the_center/Level1_Gore.png?raw=true)                | ![Stone Wall + Flying Enemies](assets/images/defend_the_center/Level2_Stone_Wall_Flying_Enemies.png?raw=true) | ![Resized Flying Enemies + Mossy Bricks](assets/images/defend_the_center/Level3_Resized_Flying_Enemies_Mossy_Bricks.png?raw=true) | ![Complete](assets/images/defend_the_center/Level4_Complete.png?raw=true) |
| Health Gathering  | ![Default](assets/images/health_gathering/Level0_Default.png?raw=true)  | ![Resized Kits](assets/images/health_gathering/Level1_Resized_Kits.png?raw=true) | ![Stone Wall + Flying Enemies](assets/images/health_gathering/Level2_Slime_Obstacles.png?raw=true)            | ![Lava + Supreme + Resized Agent](assets/images/health_gathering/Level3_Lava_Supreme_Resized_Agent.png?raw=true)                  | ![Complete](assets/images/health_gathering/Level4_Complete.png?raw=true)  |
| Seek and Slay     | ![Default](assets/images/seek_and_slay/Level0_Default.png?raw=true)     | ![Shadows](assets/images/seek_and_slay/Level1_Shadows.png?raw=true)              | ![Obstacles + Resized Enemies](assets/images/seek_and_slay/Level2_Obstacles_Resized_Enemies.png?raw=true)     | ![Red + Obstacles + Invulnerable](assets/images/seek_and_slay/Level3_Red_Obstacles_Invulnerable.png?raw=true)                     | ![Complete](assets/images/seek_and_slay/Level4_Complete.png?raw=true)     |
| Dodge Projectiles | ![Default](assets/images/dodge_projectiles/Level0_Default.png?raw=true) | ![Barons](assets/images/dodge_projectiles/Level1_Barons.png?raw=true)            | ![Revenants](assets/images/dodge_projectiles/Level2_Revenants.png?raw=true)                                   | ![Flames + Flaming Skulls + Mancubus](assets/images/dodge_projectiles/Level3_Flames_Flaming_Skulls_Mancubus.png?raw=true)         | ![Complete](assets/images/dodge_projectiles/Level4_Complete.png?raw=true) |

## Quick Start
```bash
$ python LevDoom/levdoom/run.py --algorithm ppo --scenario defend_the_center
```

## Baseline experiments

[//]: # (### Results)
[//]: # (TODO)

### Instructions for reproduction
- PPO
```bash
$ python LevDoom/levdoom/run.py --algorithm ppo --scenario $SCENARIO \
        --tasks $TASKS --test_tasks $TEST_TASKS --seed $SEED --epoch 100
```
- Rainbow
```bash
$ python LevDoom/levdoom/run.py --algorithm rainbow --scenario $SCENARIO \
        --tasks $TASKS --test_tasks $TEST_TASKS --seed $SEED --epoch 100 \ 
        --lr 0.0001 --step-per-collect 10 --batch-size 64
```
- DQN
```bash
$ python LevDoom/levdoom/run.py --algorithm dqn --scenario $SCENARIO \
        --tasks $TASKS --test_tasks $TEST_TASKS --seed $SEED --epoch 100 \ 
        --lr 0.0001 --step-per-collect 10 --batch-size 64
```

#### Arguments

`$SEED = {1, 2, 3, 4, 5}`

Use the following scenario specific arguments

1. Defend the Center
```
$SCENARIO = defend_the_center
$TASKS = default gore stone_wall fast_enemies mossy_bricks fuzzy_enemies flying_enemies resized_enemies
$TEST_TASKS = stone_wall_flying_enemies resized_fuzzy_enemies resized_flying_enemies_mossy_bricks gore_stone_wall_fuzzy_enemies gore_mossy_bricks fast_resized_enemies_gore complete
```
2. Health Gathering
```
$SCENARIO = health_gathering
$TASKS = default lava slime supreme poison obstacles stimpacks shaded_kits resized_kits
$TEST_TASKS = supreme_poison slime_obstacles shaded_stimpacks poison_resized_shaded_kits obstacles_slime_stimpacks lava_supreme_resized_agent complete
```
3. Dodge Projectiles
```
--scenario dodge_projectiles \
--tasks default cacodemons barons city flames mancubus resized_agent flaming_skulls \
--test_tasks city_resized_agent revenants barons_flaming_skulls city_arachnotron flames_flaming_skulls_mancubus
```
4. Seek and Slay
```
--scenario seek_and_slay \
--tasks default red blue shadows obstacles invulnerable mixed_enemies resized_enemies \
--test_tasks blue_shadows obstacles_resized_enemies invulnerable_blue blue_mixed_resized_enemies red_obstacles_invulnerable resized_shadows_red complete
```

#### WandB support

LevDoom also supports experiment monitoring with Weights and Biases. In order to setup WandB locally
run `wandb login` in the terminal ([WandB Quickstart](https://docs.wandb.ai/quickstart#1.-set-up-wandb)).

Example command line to run an experiment with WandB monitoring:

```bash
$ python LevDoom/levdoom/run.py --scenario health_gathering --algorithm ppo --with_wandb True \ 
        --wandb_user <your_wandb_user> --wandb_key <your_wandb_api_key> --wandb_tags benchmark doom ppo
```

A total list of WandB settings:

```
--with_wandb: Enables Weights and Biases integration (default: False)
--wandb_user: WandB username (entity). Must be specified from command line! Also see https://docs.wandb.ai/quickstart#1.-set-up-wandb (default: None)
--wandb_key: WandB API key. Might need to be specified if running from a remote server. (default: None)
--wandb_project: WandB "Project" (default: LevDoom)
--wandb_group: WandB "Group" (to group your experiments). By default this is the name of the env. (default: None)
--wandb_job_type: WandB job type (default: SF)
--wandb_tags: [WANDB_TAGS [WANDB_TAGS ...]] Tags can help with finding experiments in WandB web console (default: [])
```

Once the experiment is started the link to the monitored session is going to be available in the logs (or by searching in Wandb Web console).

## Reference
Implementation of DQN, Rainbow and PPO from https://github.com/thu-ml/tianshou

## Citation
Cite as
```bib
@inproceedings{tomilin2022levdoom,
  title     = {LevDoom: A Benchmark for Generalization on Level Difficulty in Reinforcement Learning},
  author    = {Tristan Tomilin and Tianhong Dai and Meng Fang and Mykola Pechenizkiy},
  booktitle = {In Proceedings of the IEEE Conference on Games},
  year      = {2022}
}
```