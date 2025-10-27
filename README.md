# How to run
## Create virtual environment
In the DRL-Testing-Games directory:
```
python -m venv venv  
source venv/bin/activate  # Linux/Mac  
.venv\Scripts\activate    # Windows
```
## Install dependencies
```
pip install -r requirements.txt
```
## Agent Training

### Maze Game
```
python -m src.train_maze --algo a2c --persona survivor --timesteps 100000

python -m src.train_maze --algo a2c --persona explorer --timesteps 100000

python -m src.train_maze --algo ppo --persona survivor --timesteps 100000

python -m src.train_maze --algo ppo --persona explorer --timesteps 100000
```

### Subway Sufers Like Game
```
python -m src.train_subway --algo a2c --persona survivor --timesteps 100000

python -m src.train_subway --algo a2c --persona explorer --timesteps 100000

python -m src.train_subway --algo ppo --persona survivor --timesteps 100000

python -m src.train_subway --algo ppo --persona explorer --timesteps 100000
```

## Evaluate a Model
Use any combination of algo (ppo, a2c) and persona (survivor, explorer) that you have trained in the following format:
### Maze
```
python -m src.eval_maze --algo a2c --persona explorer --episodes 10
```
Use --render to view the model play one live visualization episode
### Subway
```
python -m src.eval_subway --algo ppo --persona survivor --episodes 10
```

## Environments
| Environment       | Observation Space                                         | Action Space                      | Rewards / Persona                                                                 |
| ----------------- | --------------------------------------------------------- | --------------------------------- | --------------------------------------------------------------------------------- |
| **Subway** | 1D lane position + obstacles                              | 0: stay, 1: left, 2: right        | Survivor: +1 per step alive, -1 for crash; Explorer: bonus for visiting new lanes |
| **Maze**     | 5×5 grid with agent=1, goal=0.5, obstacles=-1 | 0: up, 1: down, 2: left, 3: right | Survivor: -0.01 per step, +1 for goal; Explorer: +0.1 per new tile, +1 for goal   |

### Example Obseravtion space for the Maze Game:
```
[[ 1.  0.  0. -1.  0.]  
 [ 0. -1.  0.  0.  0.]  
 [ 0.  0.  0.  0.  0.]  
 [ 0.  0. -1.  0.  0.]  
 [ 0.  0.  0.  0.  0.5]]
```
