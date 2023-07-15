# these are ordered dicts where the key : value
# is env_name : env_constructor
import argparse
import os
import pickle
import shutil
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)
from metaworld.policies import *

all_v2_pol_instance = {'assembly-v2-goal-observable': SawyerAssemblyV2Policy(),
                       'basketball-v2-goal-observable': SawyerBasketballV2Policy(),
                       'bin-picking-v2-goal-observable': SawyerBinPickingV2Policy(),
                       'box-close-v2-goal-observable': SawyerBoxCloseV2Policy(),
                       'button-press-topdown-v2-goal-observable': SawyerButtonPressTopdownV2Policy(),
                       'button-press-topdown-wall-v2-goal-observable': SawyerButtonPressTopdownWallV2Policy(),
                       'button-press-v2-goal-observable': SawyerButtonPressV2Policy(),
                       'button-press-wall-v2-goal-observable': SawyerButtonPressWallV2Policy(),
                       'coffee-button-v2-goal-observable': SawyerCoffeeButtonV2Policy(),
                       'coffee-pull-v2-goal-observable': SawyerCoffeePullV2Policy(),
                       'coffee-push-v2-goal-observable': SawyerCoffeePushV2Policy(),
                       'dial-turn-v2-goal-observable': SawyerDialTurnV2Policy(),
                       'disassemble-v2-goal-observable': SawyerDisassembleV2Policy(),
                       'door-close-v2-goal-observable': SawyerDoorCloseV2Policy(),
                       'door-lock-v2-goal-observable': SawyerDoorLockV2Policy(),
                       'door-open-v2-goal-observable': SawyerDoorOpenV2Policy(),
                       'door-unlock-v2-goal-observable': SawyerDoorUnlockV2Policy(),
                       'hand-insert-v2-goal-observable': SawyerHandInsertV2Policy(),
                       'drawer-close-v2-goal-observable': SawyerDrawerCloseV2Policy(),
                       'drawer-open-v2-goal-observable': SawyerDrawerOpenV2Policy(),
                       'faucet-open-v2-goal-observable': SawyerFaucetOpenV2Policy(),
                       'faucet-close-v2-goal-observable': SawyerFaucetCloseV2Policy(),
                       'hammer-v2-goal-observable': SawyerHammerV2Policy(),
                       'handle-press-side-v2-goal-observable': SawyerHandlePressSideV2Policy(),
                       'handle-press-v2-goal-observable': SawyerHandlePressV2Policy(),
                       'handle-pull-side-v2-goal-observable': SawyerHandlePullSideV2Policy(),
                       'handle-pull-v2-goal-observable': SawyerHandlePullV2Policy(),
                       'lever-pull-v2-goal-observable': SawyerLeverPullV2Policy(),
                       'peg-insert-side-v2-goal-observable': SawyerPegInsertionSideV2Policy(),
                       'pick-place-wall-v2-goal-observable': SawyerPickPlaceWallV2Policy(),
                       'pick-out-of-hole-v2-goal-observable': SawyerPickOutOfHoleV2Policy(),
                       'reach-v2-goal-observable': SawyerReachV2Policy(),
                       'push-back-v2-goal-observable': SawyerPushBackV2Policy(),
                       'push-v2-goal-observable': SawyerPushV2Policy(),
                       'pick-place-v2-goal-observable': SawyerPickPlaceV2Policy(),
                       'plate-slide-v2-goal-observable': SawyerPlateSlideV2Policy(),
                       'plate-slide-side-v2-goal-observable': SawyerPlateSlideSideV2Policy()
    , 'plate-slide-back-v2-goal-observable': SawyerPlateSlideBackV2Policy(),
                       'plate-slide-back-side-v2-goal-observable': SawyerPlateSlideBackSideV2Policy(),
                       'peg-unplug-side-v2-goal-observable': SawyerPegUnplugSideV2Policy(),
                       'soccer-v2-goal-observable': SawyerSoccerV2Policy(),
                       'stick-push-v2-goal-observable': SawyerStickPushV2Policy(),
                       'stick-pull-v2-goal-observable': SawyerStickPullV2Policy(),
                       'push-wall-v2-goal-observable': SawyerPushWallV2Policy(),
                       'reach-wall-v2-goal-observable': SawyerReachWallV2Policy(),
                       'shelf-place-v2-goal-observable': SawyerShelfPlaceV2Policy(),
                       'sweep-into-v2-goal-observable': SawyerSweepIntoV2Policy(),
                       'sweep-v2-goal-observable': SawyerSweepV2Policy(),
                       'window-open-v2-goal-observable': SawyerWindowOpenV2Policy(),
                       'window-close-v2-goal-observable': SawyerWindowCloseV2Policy()}

all_view = ('corner', 'corner2', 'corner3', 'topview')


def zip_compression(source_dir, target_file):
    with ZipFile(target_file, mode='w') as zf:
        for path, dir_names, filenames in os.walk(source_dir):
            path = Path(path)
            arc_dir = path.relative_to(source_dir)
            for filename in filenames:
                zf.write(path.joinpath(filename), arc_dir.joinpath(filename))


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def main(index):
    try:
        env_keys = list(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys())
        print(env_keys)
        pol_and_env_num = len(all_v2_pol_instance)
        sample_num = 500
        num_eposides = 1

        env_key = env_keys[index]

        policy = all_v2_pol_instance[env_key]

        for k in range(num_eposides):
            env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_key]()
            obs = env.reset_model()
            obs = env.reset()
            # print()
            param_list = []
            save_path = f"./metaworld_data/{env_key}/{env_key}_{k}"
            mkdir(save_path)
            param_save_path = os.path.join(save_path, 'param')
            mkdir(param_save_path)
            rgb_save_path = os.path.join(save_path, 'rgb')
            mkdir(rgb_save_path)

            for j in range(sample_num):

                a = policy.get_action(obs)
                obs, reward, done, info = env.step(a)
                # other param
                param_dict = {
                    'action': a,
                    'reward': reward,
                    'obs': obs,
                    'is_done': done,
                    'info': info,
                }
                param_list.append(param_dict)

                # rgb
                for view in all_view:
                    rgb_view_save_path = os.path.join(rgb_save_path, view)
                    mkdir(rgb_view_save_path)
                    rgb = env.render('rgb_array', camera_name=view, resolution=(256, 256))
                    p = Image.fromarray(rgb)
                    img_save_path = os.path.join(rgb_view_save_path, f"image_save_{j}.jpg")
                    p.save(img_save_path)

                if info['success'] > 0:
                    pickle_save_path = os.path.join(param_save_path, 'param.pickle')
                    with open(pickle_save_path, 'wb') as f:
                        pickle.dump(param_list, f)
                    zip_compression(save_path, save_path + '.zip')

                    target_save_path = f"./metaworld_data/{env_key}/{env_key}_{k}.zip"
                    mkdir(f"./metaworld_data/{env_key}")
                    shutil.move(save_path + '.zip', target_save_path)
                    shutil.rmtree(save_path, ignore_errors=True)
                    break

                if j == 499:
                    shutil.rmtree(save_path)

    except (Exception) as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--tasks', type=int, default=50)
    args = parser.parse_args()
    process_list = []
    pol_and_env_num = len(all_v2_pol_instance)
    for k in range(0, args.tasks):
        main(k)
