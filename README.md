# BATON: Biomechanics-Aware Trajectory Optimization for Navigation during robotic physiotherapy
BATON is a novel method for planning biomechanics-aware robotic rehabilitation trajectories for rotator cuff therapy. It embeds a human musculoskeletal model into the robotic controller to plan trajectories that achieve a desired final human pose while minimizing strain in selected tendons and accounting for the dynamics of the human arm with which the robot interacts.

<img src="Media/visual_abstract.png" height="300" />

Our approach is presented in detail in:

```bib
@article{belli2024biomechanics,
  title={Biomechanics-Aware Trajectory Optimization for Navigation during Robotic Physiotherapy},
  author={Belli, Italo and Prendergast, J Micah and Seth, Ajay and Peternel, Luka},
  journal={arXiv preprint arXiv:2411.03873},
  year={2024}
}
```


<table align="center">
  <tr>
    <td colspan="2" align="center">Funding Institutions</td>
  </tr>
  <tr>
    <td align="center">
      <a>
        <img src="https://user-images.githubusercontent.com/50029203/226883398-97b28065-e144-493b-8a6c-5cbbd9000411.png" alt="TUD logo" height="128">
        <br />
        <a href="https://www.tudelft.nl/3me/over/afdelingen/cognitive-robotics-cor">Cognitive Robotics</a> and <br />
        <a href="https://www.tudelft.nl/3me/over/afdelingen/biomechanical-engineering">Biomechanical Engineering</a> at TU Delft</p>
      </a>
    </td>
    <td align="center">
      <a href="https://chanzuckerberg.com/">
        <img src="https://user-images.githubusercontent.com/50029203/226883506-fbb59348-38a4-43f9-93c9-2c7b8ba63619.png" alt="CZI logo" width="128" height="128">
        <br />
        Chan Zuckerberg Initiative
      </a>
    </td>
  </tr>
</table>

## Features
When rehabilitating from rotator cuff tears, physiotherapy aims at gaining a large range of motion while avoiding injuries to the healing tendon(s). A robot that interacts with patients at this stage needs to have insights into the inner functioning of the human tissues. Our innovation consists in considering a [state-of-the-art biomechanical model of the human shoulder](https://simtk.org/projects/scapulothoracic) to extract:
- _muscle activations_ during physical human-robot interaction (through a model-based [rapid muscle redundancy solver](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0295003));
- _strain maps_, which give insights into how tendons are elongated during the rehabilitation movement (as a consequence of the position of the patient, and of the activation in the corresponding muscles);
- _skeletal dynamics_, capturing the way in which the human position evolves as a function of torques applied to the human model.

These elements are combined in an optimal control problem that can be solved efficiently in 0.12 s over a time horizon of 1 s divided into 10 steps.

To efficiently consider human skeletal dynamics, we exploit a [customized version](https://github.com/itbellix/opensimAD) of [OpenSimAD](https://github.com/antoinefalisse/opensimAD) that allows us to retrieve a differentiable expression for the dynamics of the original OpenSim model, and can be natively interfaced with [CasADi](https://web.casadi.org/).

## Requirements
Our code has been tested on Ubuntu 20.04.

In order to run BATON you will need:
- an OpenSimAD Conda environment to retrieve your differentiable OpenSim model. This can be set up following the instructions at https://github.com/antoinefalisse/opensimAD
- OpenSim itself, available either through the official [Conda Package](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53116061/Conda+Package) or (preferably) a [local build](https://github.com/opensim-org/opensim-core/blob/main/scripts/build/opensim-core-linux-build-script.sh)
- a ROS distribution (we tested our code with [ROS Noetic](http://wiki.ros.org/noetic));
- a working version of our [`iiwa-impedance-control`](https://gitlab.tudelft.nl/nickymol/iiwa_impedance_control) package, implementing the impedance controller for the KUKA LBR iiwa collaborative robotic arm that we used in our experiments. Please refer to the project page for obtaining and building it correctly. Note that this repository has quite a few dependency, so make sure to follow the [set-up instructions](https://gitlab.tudelft.nl/nickymol/iiwa_impedance_control/-/tree/set_up_instructions?ref_type=heads).


In terms of sensing hardware, we used the force/torque sensor Bota SenseONE to monitor contact wrenches between the human and the Kuka robot.

## Structure
Our code is organized as follows:

- [`launch`](https://github.com/itbellix/baton-robotic-rehab/tree/main/launch) contains the launch file to start the ROS master and take care of bringing up the impedance controller modules, or Gazebo for simulation;
- [`scripts`](https://github.com/itbellix/baton-robotic-rehab/tree/main/scripts) contains the Python scripts specific to run the various modules of `BATON` and reproduce our figures

- [`Musculoskeletal Models`](<https://github.com/itbellix/baton-robotic-rehab/tree/main/Musculoskeletal Models>): this folder contains the OpenSim models that we used with `BATON`. In particular, a reduced-order version of the [`thoracoscapular shoulder model`](https://simtk.org/projects/scapulothoracic) is considered, capturing the mobility of the glenohumeral joint and the musculo-tendon units spanning the joint.

Further explanations are provided inside each folder.


## Brief guide to our code
In order to run our code, you will need 4 different terminals (I choose to keep things separate, but a ROS package + launchfile would be cleaner). 
On all of them, navigate to your local version of this repository, and source your ROS distribution, and the catkin workspace containing `iiwa-impedance-control`. Then, assuming that you are running things in simulation:
- on _terminal-1_: run `roslaunch Code/launch/bringup.launch simulation:=true`

- then, on the other terminals, activate a virtual environment created on the basis of the [requirements.txt](https://github.com/itbellix/baton-robotic-rehab/blob/main/requirements.txt), and then:
  - on _terminal-2_: run `python Code/scripts/robot_control.py --simulation=true`
  - on _terminal-3_: run `python Code/scripts/TO_main.py --simulation=true`
  - on _terminal-4_: run `python estimate_muscle_activation.py --simulation=true`


If everything was installed correctly, you will see the Gazebo environment with the Kuka robot, and you should be prompted with a selection menu as below:

<img src="Media/selection_menu.png" height="150" />

Then, you can input `a` so that the robot moves to the starting position for the therapy.
Once the position has been reached, you will see a strain map being brought up on the screen, and you will be able to input `s` so that the simulated experiment can start. By default, experiment 1 will be executed (where we assume position-dependent strains in the rotator cuff tendons). This can be changed in `scripts/experimental_parameters.py`.

Overall, the windows that you will see should look like, displaying the generation of rehabilitation trajectories that minimize the strain on the rotator cuff tendons:

<img src="Media/display_baton_sim.png" height="200" />

### Trouble-shooting
- **Slow execution**: note that running everything on one machine can be quite slow. For this, in our experiments we employed an external workstation to run the robot controller (_terminal-1_ above), and to run a `rqt` GUI for visualization of contact wrenches and estimated muscle activations. Then, a normal laptop was used to run the rest of the scripts. Communication between the two machines and the robot was established with a Netgear switch.


If you encounter any troubles or issues with running the code contained in this repository, feel free to open an issue and report it. We will do our best to help you!

## License
Our code is licensed under the Apache 2.0 license (see the `LICENSE_code` file), while the data and models are licensed under CC BY 4.0 Use Agreement terms.
```
Technische Universiteit Delft hereby disclaims all copyright interest in the program “BATON: Biomechanics-Aware Trajectory Optimization for Navigation during robotic physiotherapy”
developed by the Author(s).

Prof. Dr. Ir. Fred van Keulen, Dean of Faculty of Mechanical, Maritime and Materials Engineering (3mE).
```

## Contributors
Italo Belli, Florian van Melis
