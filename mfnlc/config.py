from os.path import dirname, abspath

import mfnlc

default_device = "cuda"
ROOT = dirname(abspath(mfnlc.__file__))

env_config = {
    "Nav": {
        "max_step": 200,
        "goal_dim": 2,
        "state_dim": 0,
        "sink": [0, 0],
        "difficulty": {
            # number of obstacle | map size
            1: [8, [[-1, -1], [1, 1]]],
            2: [32, [[-2, -2], [2, 2]]],
            3: [128, [[-4, -4], [4, 4]]]
        }
    },
    "Point": {
        "max_step": 200,
        "goal_dim": 2,
        "state_dim": 0,
        "robot_radius": 0.3,
        "sink": [0.0, 0.0, 0.0, 0.0, 9.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "env_prop": {
            "vases_num": 0,
            "render_lidar_markers": False,
            'floor_display_mode': True
        },
        "difficulty": {
            # number of obstacle | map size
            1: [8, [[-2, -2], [2, 2]]],
            2: [32, [[-4, -4], [4, 4]]],
            3: [128, [[-8, -8], [8, 8]]]
        }
    },
    "Car": {
        "max_step": 1000,
        "goal_dim": 2,
        "state_dim": 0,
        "robot_radius": 0.3,
        "sink": [0.0, 0.0, 0.0, 0.0, 9.8, 0.0, 0.0, 0.0, 1.0,
                 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "env_prop": {
            "vases_num": 0,
            "render_lidar_markers": False,
            'floor_display_mode': True
        },
        "difficulty": {
            # number of obstacle | map size
            1: [8, [[-2, -2], [2, 2]]],
            2: [32, [[-4, -4], [4, 4]]],
            3: [128, [[-8, -8], [8, 8]]]
        }
    },
    "Doggo": {
        "max_step": 1000,
        "goal_dim": 2,
        "state_dim": 0,
        "robot_radius": 0.3,
        "sink": [0.0, 0.0, 0.47, 0.0, 9.8, 0.0, -0.0, 0.0, 0.01, 1.0, 0.0, 1.0, 0.0, 1.0, 0.01, 1.0, 0.26, 0.97, 0.01,
                 1.0, 0.53, 0.85, 0.01, 1.0, 0.53, 0.85, 0.01, 1.0, 0.26, 0.97, 0.01, 1.0, -0.0, 0.0, 0.0, -0.0, -0.0,
                 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.44, -0.24, -0.02, 0.0, 0.13, 0.0, 0.06, 0.0, 0.06, 0.0, 0.13,
                 -0.0, -0.0, 0.0],
        "env_prop": {
            "vases_num": 0,
            "render_lidar_markers": False,
            'floor_display_mode': True
        },
        "difficulty": {
            # number of obstacle | map size
            1: [8, [[-2, -2], [2, 2]]],
            2: [32, [[-4, -4], [4, 4]]],
            3: [128, [[-8, -8], [8, 8]]]
        }
    }
}


def get_path(robot_name, algo, task):
    data_root = f"{ROOT}/mfnlc_data"
    if task == "log":
        return f"{data_root}/{algo}/{robot_name}/{task}"
    elif task == "model":
        ext = "zip" if algo != "cpo" else "onnx"
        return f"{data_root}/{algo}/{robot_name}/model.{ext}"
    elif task == "tclf":
        return f"{data_root}/{algo}/{robot_name}/tclf.pth"
    elif task == "comparison":
        return f"{data_root}/comparison/{robot_name}"
    elif task == "evaluation":
        return f"{data_root}/{algo}/{robot_name}/evaluation"
    elif task == "lv_table":
        return f"{data_root}/lyapunov_td3/{robot_name}/lv_table.pkl"
    elif task == "video":
        return f"{data_root}/{algo}/{robot_name}/evaluation/video_"
