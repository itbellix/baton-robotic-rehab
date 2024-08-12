# MUST: musculoskeletal trajectory optimization
MUST is a novel method for planning musculoskeletal-aware robotic rehabilitation trajectories for rotator cuff therapy. It embeds a human musculoskeletal model into the robotic controller to plan trajectories that achieve a desired final human pose while minimizing strain in selected tendons and accounting for the dynamics of the human arm with which the robot interacts.

<img src="Media/visual_abstract_1.svg" height="300" />

Our approach is presented in detail in:

```bib
@article{must,
  title={MUST: musculoskeletal},
  author={Belli, Italo and Prendergast, J Micah and Seth, Ajay and Peternel, Luka},
  journal={XX},
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
- _strain maps_, which give insights into how tendons are elongated during the rehabilitation movement (as a consequence of the position of the patient, and of the activation in the corresponding muscles)
- _skeletal dynamics_, capturing the way in which the human position evolves as a function of torques applied to the human model

These two elements are combined in an optimal control problem, that can be solved efficiently in 0.12 s over a time horizon of 1 s divided in 10 steps.

We exploit a customized version of OpenSimAD that allows us to retrieve a differentiable expression for the dynamics of the original OpenSim model, and can be natively interfaced with CasADi.

## Requirements
In order to run **MUST** you will need:
- CasADi (avaible here: https://web.casadi.org/)
- an OpenSimAD Conda environment (which can be set up following the instructions at https://github.com/antoinefalisse/opensimAD)

## Structure


## Brief guide to our code

### Trouble-shooting
If you encounter any troubles or issues with running the code contained in this repository, feel free to open an issue and report it. We will do our best to help you!

## License
Our code is licensed under the Apache 2.0 license (see the `LICENSE_code` file), while the data and models are licensed under CC BY 4.0 Use Agreement terms.
```
Technische Universiteit Delft hereby disclaims all copyright interest in the program “MUST: musculoskeletal trajectory optimization”
developed to solve the muscle redundancy problem in biomechanical models written by the Author(s).

Prof. Dr. Ir. Fred van Keulen, Dean of Faculty of Mechanical, Maritime and Materials Engineering (3mE).
```

## Contributors
Italo Belli