# TrajSampler for quadrotors

Sampling-based MPC.

Test/develop with:
```
# Some debug nodes to plot data and get insights

# Some general timing
rosrun rpg_mppi debug_node

# Compare forward integration for different integration step sizes and integrators
rosrun rpg_mppi debug_integration 

# Inspect extensive input sampling & plot resulting trajectories and their costs
rosrun rpg_mppi debug_label_generation
```

Launch with simulation in the loop:
```
roslaunch rpg_mppi simulation.launch
# OR
roslaunch fpv_aggressive_trajectories simulation.launch
```

Unit test:
```
catkin run_tests -j1 rpg_mppi --no-deps
```
