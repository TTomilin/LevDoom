# GVizDoom
Platform for training generalizable deep reinforcement learning agents

## Installation
```
git clone https://github.com/TTomilin/GVizDoom
```

## Algorithms implemented
* Dueling-DDQN

## Scenarios and tasks Implemented
* Defend the Center
  * Default
  * Gore
  * Mossy Bricks
  * Stone Wall
  * Flying Enemies
  * Fast Enemies
  * Fuzzy Enemies
  * Resized Enemies
* Health Gathering
* Seek and Kill
* Dodge Projectiles

## Command line usage
#### Example
```
python3 training/main.py --algorithm dueling_ddqn --scenario defend_the_center \ 
    --tasks default gore stone_wall -observe 10000 --seed 1234 --model-save-frequency 25000
```