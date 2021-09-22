import math
import os
import shutil
import sys
import time
import datetime

import yaml


def create_settings(settings_yaml, mode='test'):
    setting_dict = {'train': TrainSetting,
                    'openloop': OpenLoopSetting,
                    'test': TestSetting,
                    'dagger': DaggerSetting}
    settings = setting_dict.get(mode, None)
    if settings is None:
        raise IOError("Unidentified Settings")
    settings = settings(settings_yaml)
    if mode == 'test' or mode == 'openloop':
        settings.freeze_backbone = True
    return settings


class Settings:
    def __init__(self, settings_yaml, generate_log=True):
        assert os.path.isfile(settings_yaml), settings_yaml

        with open(settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)

            self.quad_name = settings['quad_name']
            self.odometry_topic = settings['odometry_topic']
            self.rgb_topic = settings['rgb_topic']
            self.depth_topic = settings['depth_topic']
            # Input mode
            self.use_rgb = settings['use_rgb']
            self.use_depth = settings['use_depth']
            self.img_width = settings['img_width']
            self.img_height = settings['img_height']
            self.future_time = settings['future_time']
            # Output config
            self.state_dim = settings['state_dim']
            self.out_seq_len = settings['out_seq_len']
            self.predict_state_number = settings['predict_state_number']
            self.modes = settings['modes']
            self.seq_len = settings['seq_len']
            # net inputs
            inputs = settings['inputs']
            self.use_position = inputs['position']
            self.use_attitude = inputs['attitude']
            self.use_bodyrates = inputs['bodyrates']
            self.velocity_frame = inputs['velocity_frame']

            # --- checkpoint ---
            checkpoint = settings['checkpoint']
            self.resume_training = checkpoint['resume_training']
            assert isinstance(self.resume_training, bool)
            self.resume_ckpt_file = checkpoint['resume_file']

            # Save a copy of the parameters for reproducibility
            log_root = settings['log_dir']
            if not log_root == '' and generate_log:
                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                self.log_dir = os.path.join(log_root, current_time)
                os.makedirs(self.log_dir)
                net_file = "./src/PlannerLearning/models/nets.py"
                assert os.path.isfile(net_file)
                shutil.copy(net_file, self.log_dir)
                shutil.copy(settings_yaml, self.log_dir)

    def add_flags(self):
        self._add_flags()

    def _add_flags(self):
        raise NotImplementedError


class TrainSetting(Settings):
    def __init__(self, settings_yaml):
        super(TrainSetting, self).__init__(settings_yaml, generate_log=True)
        self.settings_yaml = settings_yaml
        self.add_flags()

    def _add_flags(self):
        with open(self.settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)
            # --- Train Time --- #
            train_conf = settings['train']
            self.max_training_epochs = train_conf['max_training_epochs']
            self.data_save_freq = train_conf['data_save_freq']
            self.batch_size = train_conf['batch_size']
            self.summary_freq = train_conf['summary_freq']
            self.train_dir = train_conf['train_dir']
            if not os.path.isdir(self.train_dir):
                os.makedirs(self.train_dir)
            self.val_dir = train_conf['val_dir']
            self.top_trajectories = train_conf['top_trajectories']
            self.log_images = train_conf['log_images']
            self.freeze_backbone = train_conf['freeze_backbone']
            assert isinstance(self.log_images, bool)
            self.save_every_n_epochs = train_conf['save_every_n_epochs']
            self.ref_frame = train_conf['ref_frame']
            assert (self.ref_frame == 'bf') or (self.ref_frame == 'wf')
            self.track_global_traj = train_conf['track_global_traj']


class TestSetting(Settings):
    def __init__(self, settings_yaml):
        super(TestSetting, self).__init__(settings_yaml, generate_log=True)
        self.settings_yaml = settings_yaml
        self.add_flags()

    def _add_flags(self):
        with open(self.settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)
            self.ref_frame = settings['ref_frame']
            # special input
            inputs = settings['inputs']
            self.pitch_angle = inputs['pitch_angle'] / 180.0 * math.pi
            test_time = settings['test_time']
            self.execute_nw_predictions = test_time['execute_nw_predictions']
            self.perform_inference = test_time['perform_inference']
            assert isinstance(self.execute_nw_predictions, bool)
            self.max_rollouts = test_time['max_rollouts']
            self.expert_folder = test_time['expert_folder']
            self.crashed_thr = test_time['crashed_thr']
            # Prediction speed
            self.network_frequency = test_time['network_frequency']
            self.fallback_radius_expert = test_time['fallback_radius_expert']
            self.accept_thresh = test_time['accept_thresh']
            self.input_update_freq = test_time['input_update_freq']
            # spacings
            self.tree_spacings = test_time['spacings']
            self.verbose = settings['verbose']
            assert isinstance(self.verbose, bool)
            self.track_global_traj = test_time['track_global_traj']
            # Unity
            unity = settings['unity']
            self.unity_start_pos = unity['unity_start_pos']
            self.random_seed = unity['random_seed']



class OpenLoopSetting(Settings):
    def __init__(self, settings_yaml):
        super(OpenLoopSetting, self).__init__(settings_yaml, generate_log=True)
        self.settings_yaml = settings_yaml
        self.add_flags()

    def _add_flags(self):
        with open(self.settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)
            # --- Train Time --- #
            test_conf = settings['test']
            self.data_save_freq = test_conf['data_save_freq']
            self.batch_size = test_conf['batch_size']
            self.test_dir = test_conf['test_dir']
            self.mode = test_conf['mode']
            assert self.mode == 'loss' or self.mode == 'prediction', "Wrong testing mode"
            self.top_trajectories = test_conf['top_trajectories']
            self.ref_frame = test_conf['ref_frame']
            assert (self.ref_frame == 'bf') or (self.ref_frame == 'wf')
            self.track_global_traj = test_conf['track_global_traj']


class DaggerSetting(Settings):
    def __init__(self, settings_yaml):
        super(DaggerSetting, self).__init__(settings_yaml, generate_log=True)
        self.settings_yaml = settings_yaml
        self.add_flags()

    def _add_flags(self):
        with open(self.settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)
            # --- Data Generation --- #
            data_gen = settings['data_generation']
            self.max_rollouts = data_gen['max_rollouts']
            self.train_every_n_rollouts = data_gen['train_every_n_rollouts']
            self.expert_folder = data_gen['expert_folder']
            self.increase_net_usage_every_n_rollouts = data_gen['increase_net_usage_every_n_rollouts']
            # --- Test Time --- #
            inputs = settings['inputs']
            self.pitch_angle = inputs['pitch_angle'] / 180.0 * math.pi
            test_time = settings['test_time']
            self.execute_nw_predictions = test_time['execute_nw_predictions']
            assert isinstance(self.execute_nw_predictions, bool)
            self.perform_inference = test_time['perform_inference']
            self.crashed_thr = test_time['crashed_thr']
            self.input_update_freq = test_time['input_update_freq']
            self.accept_thresh = test_time['accept_thresh']
            self.fallback_radius_expert = test_time['fallback_radius_expert']
            # Prediction speed
            self.network_frequency = test_time['network_frequency']
            # --- Train Time --- #
            train_conf = settings['train']
            self.max_training_epochs = train_conf['max_training_epochs']
            self.data_save_freq = train_conf['data_save_freq']
            self.batch_size = train_conf['batch_size']
            self.top_trajectories = train_conf['top_trajectories']
            self.log_images = train_conf['log_images']
            self.freeze_backbone = train_conf['freeze_backbone']
            self.summary_freq = train_conf['summary_freq']
            self.train_dir = train_conf['train_dir']
            if not os.path.isdir(self.train_dir):
                os.makedirs(self.train_dir)
            self.val_dir = train_conf['val_dir']
            self.save_every_n_epochs = train_conf['save_every_n_epochs']
            self.ref_frame = train_conf['ref_frame']
            assert (self.ref_frame == 'bf') or (self.ref_frame == 'wf')
            self.track_global_traj = train_conf['track_global_traj']
            # spacings
            self.tree_spacings = train_conf['spacings']
            self.verbose = settings['verbose']
            assert isinstance(self.verbose, bool)
            # Unity
            unity = settings['unity']
            self.unity_start_pos = unity['unity_start_pos']
            self.random_seed = unity['random_seed']
