Example Vision
==============

Render images by using rpg_flightmare. 

1) `export RPGQ_PARAM_DIR=~/catkin_plan/src/rpg_flightmare` (path to flightmare)
2) launch flightmare build (find latest release here: `https://github.com/uzh-rpg/rpg_flightmare/releases`) 
3a) if not done yet source `~/catkin_plan/devel/setup.bash`
3b) `roslaunch example_vision camera_control.launch`
4) images are saved in `RPGQ_PARAM_DIR/examples/example_vision/src/saved_image/`