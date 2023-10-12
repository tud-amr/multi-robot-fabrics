# multi-robot-fabrics

Implementation of multi-robot-fabrics presented in our MRS 2023 paper **"Multi-Robot Local Motion Planning Using Dynamic Optimization Fabrics"**.

The current version of the paper can be cited using the following reference:
```bibtex
add ref
```

A **video** showcasing the presented approach can be found [here](https://www.youtube.com/@amrlab).


## Teaser
<img src="assets/video_rf_cv_2robots.gif" alt="2 Robots applying RF">

## Options
This repository includes examples of the application of multi-robot fabrics to point robots and Panda robotic arms.
The examples can be run 
1) without rollouts (in the paper referred to as MRDF),
2) with rollout fabrics and deadlock resolution heuristic (in the paper referred to as RF), and
3) with rollout fabrics, constant velocity goal estimation and deadlock resolution (in the paper referred to as RF-CV)

While in the paper dynamic fabrics ([Spahn2023](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10086617)), are applied we also support static fabrics as introduced in [Ratliff2020](https://arxiv.org/pdf/2008.02399.pdf).

Which configuration is used can be accessed in `examples/configs`. Here, also the number of robots can be adapted.

<table>
 <tr>
  <td> Point Robot </td>
  <td> 2 Panda Scenario  </td>
  <td> 3 Panda Scenario </td>
 </tr>
 <tr>
  <td> <img src="/assets/4pointmasses.png" width="250"/> </td>
  <td> <img src="/assets/2panda_scenario.png" width="250"/> </td>  
  <td> <img src="/assets/3panda_scenario.png" width="250"/> </td>
 </tr>
</table>

## Installations
Clone this repository and go to its root:

    git clone git@github.com:tud-amr/multi-robot-fabrics.git
    cd multi-robot-fabrics
    
You can install the package using poetry. For more details on poetry see [installation instructions](docs/installation.md).

    install poetry

Requirements can be found in [pyproject.toml](pyproject.toml). 

## Usage
Enter the virtual environment using:

    poetry shell
    
In the folder `multi_robot_fabrics` run

    python examples/<example-file-name>

E.g. to run the panda example `python examples/panda_panda_multifabrics.py`.
    
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

## Troubleshooting

If you run into problems of any kind or have a question, do not hesitate to open an [issue](https://github.com/tud-amr/multi-robot-fabrics/issues) on this repository. 
