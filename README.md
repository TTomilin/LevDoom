# LevDoom
LevDoom is a benchmark with difficulty levels based on visual modifications, intended for research in generalization of deep reinforcement learning agents. The benchmark is based upon [ViZDoom](https://github.com/mwydmuch/ViZDoom), a platform addressed to pixel based learning in the FPS game domain.

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
## Environment

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

## Example Usage
```bash
$ python LevDoomLab/levdoom/run.py --algorithm ppo --scenario defend_the_center
```

## Baseline experiments

### Results
TODO

### Instructions for reproduction
1. Defend the Center

- PPO
```bash
python -u LevDoomLab/levdoom/run.py --algorithm ppo --scenario defend_the_center \
        --tasks default gore stone_wall fast_enemies mossy_bricks fuzzy_enemies flying_enemies resized_enemies \
        --test_tasks stone_wall_flying_enemies resized_fuzzy_enemies resized_flying_enemies_mossy_bricks gore_stone_wall_fuzzy_enemies \
        gore_mossy_bricks fast_resized_enemies_gore complete --seed 1 --epoch 100
```
- Rainbow
```bash
python -u LevDoomLab/levdoom/run.py --algorithm rainbow --scenario defend_the_center \
        --tasks default gore stone_wall fast_enemies mossy_bricks fuzzy_enemies flying_enemies resized_enemies \
        --test_tasks stone_wall_flying_enemies resized_fuzzy_enemies resized_flying_enemies_mossy_bricks gore_stone_wall_fuzzy_enemies \
        gore_mossy_bricks fast_resized_enemies_gore complete --seed 1 --epoch 100 --buffer-size 100000 --lr 0.0001 \
         --step-per-collect 10 --batch-size 64
```
- DQN
```bash
python -u LevDoomLab/levdoom/run.py --algorithm dqn --scenario defend_the_center \
        --tasks default gore stone_wall fast_enemies mossy_bricks fuzzy_enemies flying_enemies resized_enemies \
        --test_tasks stone_wall_flying_enemies resized_fuzzy_enemies resized_flying_enemies_mossy_bricks gore_stone_wall_fuzzy_enemies \
        gore_mossy_bricks fast_resized_enemies_gore complete --seed 1 --epoch 100 --buffer-size 100000 --lr 0.0001 \
         --step-per-collect 10 --batch-size 64
```
2. Health Gathering
```bash
python -u LevDoomLab/levdoom/run.py --algorithm ppo --scenario health_gathering \
         --tasks default lava slime supreme poison obstacles stimpacks shaded_kits resized_kits \
         --test_tasks supreme_poison slime_obstacles shaded_stimpacks poison_resized_shaded_kits obstacles_slime_stimpacks \
         lava_supreme_resized_agent complete --seed 1 --epoch 100
         
 python -u LevDoomLab/levdoom/run.py --algorithm rainbow --scenario health_gathering \
         --tasks default lava slime supreme poison obstacles stimpacks shaded_kits resized_kits \
         --test_tasks supreme_poison slime_obstacles shaded_stimpacks poison_resized_shaded_kits obstacles_slime_stimpacks \
         lava_supreme_resized_agent complete --seed 1 --epoch 100 --buffer-size 100000 --lr 0.0001 \
         --step-per-collect 10 --batch-size 64
```
3. Dodge Projectiles
```bash
python -u LevDoomLab/levdoom/run.py --algorithm rainbow --scenario dodge_projectiles \
        --tasks default cacodemons barons city flames mancubus resized_agent flaming_skulls \
        --test_tasks city_resized_agent revenants barons_flaming_skulls city_arachnotron flames_flaming_skulls_mancubus \
        resized_agent_revenants complete --seed 1 --epoch 100 --buffer-size 100000 --lr 0.0001 \
        --step-per-collect 10 --batch-size 64
```
4. Seek and Slay
```bash
python -u LevDoomLab/levdoom/run.py --algorithm ppo --scenario seek_and_slay \
            --tasks default red blue shadows obstacles invulnerable mixed_enemies resized_enemies \
            --test_tasks blue_shadows obstacles_resized_enemies invulnerable_blue blue_mixed_resized_enemies \
            red_obstacles_invulnerable resized_shadows_red complete --seed 1 --epoch 100
```