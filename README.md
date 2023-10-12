# multi-robot-fabrics

Implementation of multi-robot-fabrics presented in our MRS 2023 paper "Multi-Robot Local Motion Planning Using  
Dynamic Optimization Fabrics".


Video:

## Teaser
<img src="assets/video_rf_cv_2robots.gif" alt="2 Robots applying RF">

## Explanation
This project should be set up with poetry. Installation instructions are provided in the docs folder.

This repository is meant to explore the use of fabrics for multiple mobile robots/robotic manipulators.
Several flavours are explored:
- Dynamic fabrics applied to a multirobot scenarios. 
- Rollout fabrics applied to single robot and multirobot scenarios. (Forward predictions over a horizon)

The 'examples' folder provides runable examples of different scenarios:
- panda_panda_rolloutfabrics.py explores a two robot scenario with two pandas. 
    The planner rolls out fabrics along the horizon to detect deadlocks if ROLLOUT_FABRICS=True.
    If ROLLOUT_FABRICS=False, the controller applies dynamic fabrics to the two pandas at each time-step.
- for easy understanding, also a point-robot example is illustrated in 4point_static.py and 4point_dynamic.py with 
    static and dynamic fabrics respectively.

The simulation environments are constructed in the folder 'simulation_environments' for one, two or three robots respectively.
Parameters of the two and three panda case are constructed in 'parameters_manipulators.py'.

Academic licences of forcespro can be requested via embotech.com.

## Installation
We provide detailed installation instructions [here](docs/installation.md).
