# Agile Autonomy Data Generation

This folder contains all functionality to generate trajectory labels from a simulated gazebo environment.

## Setup
1. Build [open3D](http://www.open3d.org/docs/release/compilation.html) from source (tested `0.9.0.0`). (no python bindings are needed at the moment)
2. create a new catkin workspace
3. Clone this repo
4. In the `catkin_ws/src` folder: `vcs-import < agile_autonomy/dependencies.yaml`
5. `catkin build`


## Launch Commands

To launch the simulation:
```
roslaunch agile_autonomy simulation.launch
```
