import argparse
import attr
import habitat
import numpy as np

from habitat import logger
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions

import os

from PIL import Image
import json
import glob
from zipfile import ZipFile
from pathlib import Path
import shutil

gibson_objects = [
    "null",
    "chair",
    "potted plant",
    "sink",
    "vase",
    "book",
    "couch",
    "bed",
    "bottle",
    "dining table",
    "toilet",
    "refrigerator",
    "tv",
    "clock",
    "oven",
    "bowl",
    "cup",
    "bench",
    "microwave",
    "suitcase",
    "umbrella",
    "teddy bear"
]
# start_from_zero
mp3d_objects = [
    "null",
    "chair",
    "table",
    "picture",
    "cabinet",
    "cushion",
    "sofa",
    "bed",
    "chest_of_drawers",
    "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv_monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym_equipment",
    "seating",
    "clothes"
]
hm3d_objects = [
    "null",
    "chair",
    "bed",
    "plant",
    "toilet",
    "tv_monitor",
    "sofa"
]


def crop(img):
    out = img.resize((320, 240))
    left = 40
    right = 280
    top = 0
    bottom = 240
    out = out.crop((left, top, right, bottom))
    im = out.resize((256, 256))
    return im


def zip_compression(source_dir, target_file):
    """zip dir 

    :param source_dir: 
    :param target_file:
    :return:"""
    with ZipFile(target_file, mode='w') as zf:
        for path, dir_names, filenames in os.walk(source_dir):
            path = Path(path)
            arc_dir = path.relative_to(source_dir)
            for filename in filenames:
                zf.write(path.joinpath(filename), arc_dir.joinpath(filename))


def get_episode_json(episode, reference_replay):
    episode.reference_replay = reference_replay
    episode.scene_id = episode.scene_id
    ep_json = attr.asdict(episode)
    del ep_json["_shortest_path_cache"]
    return ep_json


def save_trajectory(root_path, dir_name, pack_name, ep_id, env_dict, obs, infos, actions):
    save_path = os.path.join(root_path, dir_name, pack_name)
    json_save_path = os.path.join(save_path, ep_id, 'data.json')
    zip_save_path = os.path.join(save_path, ep_id)

    rgb_save_path = os.path.join(save_path, str(ep_id), 'rgb')
    depth_save_path = os.path.join(save_path, str(ep_id), 'depth')

    rgb_path = os.path.join(pack_name, str(ep_id), 'rgb')
    depth_path = os.path.join(pack_name, str(ep_id), 'depth')

    if not os.path.exists(rgb_save_path):
        os.makedirs(rgb_save_path)
    if not os.path.exists(depth_save_path):
        os.makedirs(depth_save_path)

    for img_number in range(len(obs)):
        observations = obs[img_number]
        info = infos[img_number]
        action = actions[img_number]

        rgb = Image.fromarray(observations['rgb'])
        rgb.save(os.path.join(rgb_save_path, str(img_number) + '.png'))

        depth = Image.fromarray((observations['depth'][:, :, 0] * 255).astype(np.uint8))
        depth.save(os.path.join(depth_save_path, str(img_number) + '.png'))

        data = {'dir_name': dir_name,
                'episode_number': ep_id,
                'img_number': int(img_number),
                'objectgoal_id': list([int(x) for x in list([0])]),
                'objectgoal': 'null',
                'gps_goal': list([float(x) for x in list(observations['pointgoal_with_gps_compass'])]),
                'point_goal': list([float(x) for x in list(observations['pointgoal'])]),
                'compass': list([float(x) for x in list(observations['compass'])]),
                'gps': list([float(x) for x in list(observations['gps'])]),
                'distance_to_goal': float(info['distance_to_goal']),
                'success': int(info['success']),
                'spl': float(info['spl']),
                'action': int(action),
                'rgb': os.path.join(rgb_path, str(img_number) + '.png'),
                'depth': os.path.join(depth_path, str(img_number) + '.png'),

                }
        env_dict[ep_id]['data'].append(data)
    with open(json_save_path, 'w') as json_file:
        json.dump(env_dict, json_file)

    if not os.path.exists(os.path.join('./temp', dir_name, pack_name)):
        os.makedirs(os.path.join('./temp', dir_name, pack_name))

    if not os.path.exists(os.path.join(root_path, dir_name, pack_name)):
        os.makedirs(os.path.join(root_path, dir_name, pack_name))

    _save_path = os.path.join('./temp', dir_name, pack_name, ep_id + '.zip')
    zip_compression(zip_save_path, _save_path)
    target_save_path = os.path.join(root_path, dir_name, pack_name, ep_id + '.zip')
    shutil.move(_save_path, target_save_path)
    shutil.rmtree(zip_save_path, ignore_errors=True)
    # shutil.remove(_save_path,ignore_errors=True)
    return target_save_path


def generate_trajectories(cfg, args):
    root_path = args.save_path
    dir_name = args.task_type
    ct = 0
    with habitat.Env(config=cfg) as env:
        goal_radius = 0.1
        spl = 0
        total_success = 0.0
        total_episodes = 0.0
        goal_radius = env.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = 0.2
        logger.info("Total episodes: {}".format(len(env.episodes)))
        json_file = []
        for _ in range(len(env.episodes)):
            follower = ShortestPathFollower(
                env._sim, goal_radius, False
            )
            observations = env.reset()
            env_dict = {}
            current_episode = env._current_episode
            episode_id = current_episode.episode_id
            scene_id = current_episode.scene_id.split('/')[-1][:-4]
            start_position = current_episode.start_position
            start_rotation = current_episode.start_rotation

            geodesic_distance = current_episode.info['geodesic_distance']

            goals = current_episode.goals[0].position

            ep_id = scene_id + "_" + str(episode_id)
            env_dict = {ep_id: {
                'data': [],
                'episode_info': {
                    'episode_id': int(episode_id),
                    'scene_id': scene_id,
                    'start_position': list([float(x) for x in list(start_position)]),
                    'start_rotation': list([float(x) for x in list(start_rotation)]),
                    'geodesic_distance': float(geodesic_distance),
                    'goals': list([float(x) for x in list(goals)])
                }
            }}
            obs = [observations]
            actions = []
            info = env.get_metrics()
            infos = []
            infos.append(info)
            success = 0
            info = {}

            while not env.episode_over:
                best_action = follower.get_next_action(
                    env.current_episode.goals[0].position

                )
                if "distance_to_goal" in info.keys() and info[
                    "distance_to_goal"] < 0.1 and best_action != HabitatSimActions.STOP:
                    best_action = HabitatSimActions.STOP

                observations = env.step(best_action)

                info = env.get_metrics()

                print(info)
                success = info["success"]
                if success > 0:
                    ct += 1
                actions.append(best_action)

                obs.append(observations)
                # print("=="*10)
                info = env.get_metrics()
                infos.append(info)

            actions.append(0)
            pack_name = scene_id

            zip_save_path = save_trajectory(root_path, dir_name, pack_name, ep_id, env_dict, obs, infos, actions)
            json_file.append([zip_save_path, len(actions), success])
            total_success += success
            spl += info["spl"]
            total_episodes += 1
        with open(root_path + "/" + dir_name + ".json", 'w') as f:
            json.dump({"data": json_file}, f)


def main(args):
    cfg = habitat.get_config(args.config_file)
    generate_trajectories(cfg, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default='data/datasets/pointnav/gibson/v2/val/val.json.gz')
    parser.add_argument('--scene_list', default='data/datasets/pointnav/gibson/v2/val/content')
    parser.add_argument('--scene_dir', default='data/scene_datasets')
    parser.add_argument('--config_file', default='tasks/pointnav_gibson.yaml')
    parser.add_argument('--save_path', default='./imagenav')
    parser.add_argument('--task_type', default='gibson')
    parser.add_argument('--scenes', type=int, default=10)

    args = parser.parse_args()

    main(args)
