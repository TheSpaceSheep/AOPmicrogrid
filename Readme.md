# AOPmicrogrid
An attempt at implementing an Adaptive Online Planning agent (Lu et al. 2019), and run it on a microgrid simulator environment.

## Current status
Testing a PPO agent on the Continual Maze environment and on the microgrid simulator

## Installation
Run the `install.sh` file

``` ./install.sh ```

Run a ppo agent on the microgrid environment using 

```
source aopmgenv/bin/activate
python main.py --agent ppo --env microgrid
```

