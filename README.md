# Neural Lyapunov Deep Reinforcement Learning
Code Repository of IROS 22' paper **Model-free Neural Lyapunov Control for Safe Robot Navigation**

[ArXiv](https://arxiv.org/abs/2203.01190) | [Demos](https://sites.google.com/view/mf-nlc)  



https://user-images.githubusercontent.com/73256697/184549907-50287e7f-fc0c-46fa-baf9-58660e8634eb.mp4


## Project Structure

```
├── README.md
├── setup.py
└── shrl
    ├── config.py       # config file, including data path, default devices, ect. 
    ├── envs            # simulation environments
    ├── evaluation      # evaluation utils
    ├── exps            # experiment scripts
    ├── learn           # low-level controller and neural Lyapunov function learning algorithms
    ├── monitor         # high-level monitor
    ├── plan            # high-level planner, RRT & RRT*
    └── tests           # test cases
```

## Quick Start
Two quick start examples:

1. Co-learning low-level controller and neural Lyapunov function  
`python exps/train/no_obstacle/lyapunov_td3/[robot-name].py`

2. Pre-compute monitor and evaluate  
`python exps/hierachical/rrt_lyapunov/[robot-name].py`

One can start tracing code from `exps` folder. 

## Bibtex
```bibtex
@inproceedings{Xiong2022ModelfreeNL,
  title={Model-free Neural Lyapunov Control for Safe Robot Navigation},
  author={Zikang Xiong and Joe Eappen and Ahmed H. Qureshi and Suresh Jagannathan},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2022},
}
```
