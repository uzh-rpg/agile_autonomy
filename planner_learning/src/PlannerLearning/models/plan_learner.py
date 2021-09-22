import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

try:
    # Ros runtime
    from .nets import create_network
    from .data_loader import create_dataset
    from .utils import MixtureSpaceLoss, TrajectoryCostLoss
except:
    # Training time
    from nets import create_network
    from data_loader import create_dataset
    from utils import MixtureSpaceLoss, TrajectoryCostLoss, convert_to_trajectory, \
            save_trajectories, transformToWorldFrame

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class PlanLearner(object):
    def __init__(self, settings):
        self.data_interface = None
        self.config = settings
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.min_val_loss = tf.Variable(np.inf,
                                        name='min_val_loss',
                                        trainable=False)

        self.network = create_network(self.config)
        self.space_loss = MixtureSpaceLoss(T=self.config.out_seq_len * 0.1, modes=self.config.modes)
        # need two instances due to pointclouds
        self.cost_loss = TrajectoryCostLoss(ref_frame=self.config.ref_frame, state_dim=self.config.state_dim)
        self.cost_loss_v = TrajectoryCostLoss(ref_frame=self.config.ref_frame, state_dim=self.config.state_dim)

        # rate scheduler
        self.learning_rate_fn = tf.keras.experimental.CosineDecayRestarts(
            			1e-3,
            			50000,
            			1.5,
            			0.75,
            			0.01)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_fn)

        self.train_space_loss = tf.keras.metrics.Mean(name='train_space_loss')
        self.val_space_loss = tf.keras.metrics.Mean(name='validation_space_loss')
        self.train_cost_loss = tf.keras.metrics.Mean(name='train_cost_loss')
        self.val_cost_loss = tf.keras.metrics.Mean(name='validation_cost_loss')

        self.global_epoch = tf.Variable(0)

        self.ckpt = tf.train.Checkpoint(step=self.global_epoch,
                                        optimizer=self.optimizer,
                                        net=self.network)

        if self.config.resume_training:
            if self.ckpt.restore(self.config.resume_ckpt_file):
                print("------------------------------------------")
                print("Restored from {}".format(self.config.resume_ckpt_file))
                print("------------------------------------------")
                return

        print("------------------------------------------")
        print("Initializing from scratch.")
        print("------------------------------------------")

    @tf.function
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.network(inputs)
            space_loss = self.space_loss(labels, predictions)
            cost_loss = self.cost_loss((inputs['roll_id'], inputs['imu'][:, -1, :12]), predictions)
            loss = space_loss + cost_loss
        gradients = tape.gradient(loss, self.network.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        self.train_space_loss.update_state(space_loss)
        self.train_cost_loss.update_state(cost_loss)
        return gradients

    @tf.function
    def val_step(self, inputs, labels):#, epoch, step):
        """
        Perform validation step.
        """

        predictions = self.network(inputs)
        space_loss = self.space_loss(labels, predictions)
        cost_loss = self.cost_loss_v((inputs['roll_id'], inputs['imu'][:, -1, :12]), predictions)
        self.val_space_loss.update_state(space_loss)
        self.val_cost_loss.update_state(cost_loss)

        return predictions

    def adapt_input_data(self, features):
        if self.config.use_rgb and self.config.use_depth:
            inputs = {"rgb": features[1][0],
                      "depth": features[1][1],
                      "roll_id": features[2],
                      "imu": features[0]}
        elif self.config.use_rgb and (not self.config.use_depth):
            inputs = {"rgb": features[1],
                      "roll_id": features[2],
                      "imu": features[0]}
        elif self.config.use_depth and (not self.config.use_rgb):
            inputs = {"depth": features[1],
                      "roll_id": features[2],
                      "imu": features[0]}
        else:
            inputs = {"imu": features[0],
                      "roll_id": features[1]}
        return inputs

    def write_train_summaries(self, features, gradients):
        with self.summary_writer.as_default():
            tf.summary.scalar('Train Space Loss', self.train_space_loss.result(),
                              step=self.optimizer.iterations)
            tf.summary.scalar('Train Traj_Cost Loss', self.train_cost_loss.result(),
                              step=self.optimizer.iterations)
            # Feel free to add more :)

    def train(self):
        print("Training Network")
        if not hasattr(self, 'train_log_dir'):
            # This should be done only once
            self.train_log_dir = os.path.join(self.config.log_dir, 'train')
            self.summary_writer = tf.summary.create_file_writer(self.train_log_dir)
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                           self.train_log_dir, max_to_keep=20)
        else:
            # We are in dagger mode, so let us reset the best loss
            self.min_val_loss = np.inf
            self.train_space_loss.reset_states()
            self.val_space_loss.reset_states()

        dataset_train = create_dataset(self.config.train_dir,
                                       self.config, training=True)
        dataset_val = create_dataset(self.config.val_dir,
                                     self.config, training=False)

        # add pointclouds to losses
        self.cost_loss.add_pointclouds(dataset_train.pointclouds)
        self.cost_loss_v.add_pointclouds(dataset_val.pointclouds)

        for epoch in range(self.config.max_training_epochs):
            # Train
            # Set learning_phase for keras (1 is train, 0 is test)
            if self.config.freeze_backbone:
                tf.keras.backend.set_learning_phase(0)
            else:
                tf.keras.backend.set_learning_phase(1)
            for k, (features, label, _) in enumerate(tqdm(dataset_train.batched_dataset)):
                features = self.adapt_input_data(features)
                gradients = self.train_step(features, label)
                if tf.equal(k % self.config.summary_freq, 0):
                    self.write_train_summaries(features, gradients)
                    self.train_space_loss.reset_states()
                    self.train_cost_loss.reset_states()
            # Eval
            tf.keras.backend.set_learning_phase(0)
            for k, (features, label, _) in enumerate(tqdm(dataset_val.batched_dataset)):
                features = self.adapt_input_data(features)
                self.val_step(features, label)
            val_space_loss = self.val_space_loss.result()
            val_cost_loss = self.val_cost_loss.result()
            validation_loss = val_space_loss + val_cost_loss
            with self.summary_writer.as_default():
                tf.summary.scalar("Validation Space Loss", val_space_loss,
                                  step=tf.cast(self.global_epoch, dtype=tf.int64))
                tf.summary.scalar("Validation Cost Loss", val_cost_loss,
                                  step=tf.cast(self.global_epoch, dtype=tf.int64))
            self.val_space_loss.reset_states()
            self.val_cost_loss.reset_states()

            self.global_epoch = self.global_epoch + 1
            self.ckpt.step.assign_add(1)

            print("Epoch: {:2d}, Val Space Loss: {:.4f}, Val Cost Loss: {:.4f}".format(
                self.global_epoch, val_space_loss, val_cost_loss))

            if validation_loss < self.min_val_loss or ((epoch + 1) % self.config.save_every_n_epochs) == 0:
                if validation_loss < self.min_val_loss:
                    self.min_val_loss = validation_loss
                if validation_loss < 10.0: # otherwise training diverged
                    save_path = self.ckpt_manager.save()
                    print("Saved checkpoint for epoch {}: {}".format(int(self.ckpt.step), save_path))

        print("------------------------------")
        print("Training finished successfully")
        print("------------------------------")

    def test(self):
        print("Testing Network")
        self.train_log_dir = os.path.join(self.config.log_dir, 'test')
        dataset_val = create_dataset(self.config.test_dir,
                                     self.config, training=False)
        self.cost_loss_v.add_pointclouds(dataset_val.pointclouds)
        if self.config.mode == 'loss':
            tf.keras.backend.set_learning_phase(0)
            for k, (features, label, _) in enumerate(tqdm(dataset_val.batched_dataset)):
                features = self.adapt_input_data(features)
                self.val_step(features, label)
            val_space_loss = self.val_space_loss.result()
            val_cost_loss = self.val_cost_loss.result()
            self.val_space_loss.reset_states()
            self.val_cost_loss.reset_states()
            print("Testing Space Loss: {:.4f} Testing Cost Loss: {:.4f}".format(val_space_loss, val_cost_loss))
        elif self.config.mode == 'prediction':

            for features, label, traj_num in tqdm(dataset_val.batched_dataset):
                features = self.adapt_input_data(features)
                prediction = self.full_post_inference(features)

                trajectories = convert_to_trajectory(label,
                                                     state=features['imu'].numpy()[:, -1, :],
                                                     config=self.config,
                                                     network=False)
                save_trajectories(folder=self.config.log_dir,
                                  trajectories=trajectories,
                                  sample_num=traj_num.numpy())

    def inference(self, inputs):
        # run time inference
        processed_pred = self.full_post_inference(inputs).numpy()
        # Assume BS = 1
        processed_pred = processed_pred[:, np.abs(processed_pred[0, :, 0]).argsort(), :]
        alphas = np.abs(processed_pred[0, :, 0])
        predictions = processed_pred[0, :, 1:]
        return alphas, predictions

    @tf.function
    def full_post_inference(self, inputs):
        predictions = self.network(inputs)
        return predictions
