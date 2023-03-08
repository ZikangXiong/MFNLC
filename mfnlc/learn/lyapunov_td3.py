import io
import os.path
import pathlib
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Iterable

import numpy as np
import torch as th
from stable_baselines3 import TD3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update, check_for_correct_spaces
from stable_baselines3.td3.policies import TD3Policy
from torch.nn import functional as F

from mfnlc.config import default_device
from mfnlc.learn.tclf import TwinControlLyapunovFunction
from mfnlc.learn.utils import list_dict_to_dict_list


class LyapunovTD3(TD3):
    """
    Co-learn the Lyapunov function and controller

    :param tclf: Twin control Lyapunov function
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param lqf_loss_cnst: The coefficient of lqf loss
    :param q_sigma: if q loss is smaller than q_sigma, begin to use lqf
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param device: device
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
            self,
            tclf: TwinControlLyapunovFunction,
            policy: Union[str, Type[TD3Policy]],
            env: Union[GymEnv, str],
            lqf_loss_cnst: float = 1.0,
            q_sigma=None,
            learning_rate: Union[float, Schedule] = 1e-3,
            buffer_size: int = 1_000_000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 100,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
            gradient_steps: int = -1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[ReplayBuffer] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            policy_delay: int = 2,
            target_policy_noise: float = 0.2,
            target_noise_clip: float = 0.5,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: str = default_device,
            _init_setup_model: bool = True,
    ):

        super(LyapunovTD3, self).__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            policy_delay=policy_delay,
            target_noise_clip=target_noise_clip,
            target_policy_noise=target_policy_noise,
            _init_setup_model=False
        )

        self.tclf = tclf
        self.tclf_optimizer = th.optim.Adam(lr=self.learning_rate, params=self.tclf.parameters())
        self.lqf_loss_cnst = lqf_loss_cnst
        self.q_sigma = q_sigma

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(LyapunovTD3, self)._setup_model()
        self.tclf_target = deepcopy(self.tclf)
        self._setup_alias()

    def _setup_alias(self):
        self.lf = self.tclf.lf
        self.lqf = self.tclf.lqf

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses, tclf_losses = [], [], []

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(
                    replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(
                    replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (
                        1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum([F.mse_loss(current_q, target_q_values)
                               for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute twin Lyapunov-control function losses
            with th.no_grad():
                current_q1 = self.critic.q1_forward(replay_data.observations,
                                                    self.actor(replay_data.observations))
            tclf_loss = self.tclf.loss(
                replay_data.observations, replay_data.actions, replay_data.next_observations,
                current_q1)
            tclf_losses.append({k: v.item() for k, v in tclf_loss.items()})

            # optimize twin Lyapunov-control function
            self.tclf_optimizer.zero_grad()
            tclf_loss["loss_sum"].backward()
            self.tclf_optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # maximize Q value
                q_loss = -self.critic.q1_forward(replay_data.observations,
                                                 self.actor(replay_data.observations)).mean()

                if self.q_sigma is None or q_loss < self.q_sigma:
                    # minimize lqf's value
                    lqf_loss = self.tclf_target.forward_lqf(replay_data.observations,
                                                            self.actor(replay_data.observations)).mean()
                    actor_loss = q_loss + self.lqf_loss_cnst * lqf_loss
                    actor_losses.append({
                        "actor_loss": actor_loss.item(),
                        "q_loss": q_loss.item(),
                        "lqf_loss": lqf_loss.item()
                    })
                else:
                    actor_loss = q_loss
                    actor_losses.append({
                        "actor_loss": actor_loss.item(),
                        "q_loss": q_loss.item(),
                    })

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.tclf.parameters(), self.tclf_target.parameters(), self.tau)

        # log
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_loss", np.mean(critic_losses))

        for k, v in list_dict_to_dict_list(tclf_losses).items():
            self.logger.record(f"train/tclf/{k}", np.mean(v))

        if len(actor_losses) > 0:
            for k, v in list_dict_to_dict_list(actor_losses).items():
                self.logger.record(f"train/actor/{k}", np.mean(v))

    def _excluded_save_params(self) -> List[str]:
        return super(LyapunovTD3, self)._excluded_save_params() + [
            "actor", "critic", "actor_target", "critic_target", "tclf", "lf", "lqf", "tclf_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer", "tclf_optimizer"]
        return state_dicts, []

    def save(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            exclude: Optional[Iterable[str]] = None,
            include: Optional[Iterable[str]] = None,
    ) -> None:
        super(LyapunovTD3, self).save(path, exclude, include)
        th.save(self.tclf, f"{os.path.dirname(path)}/tclf.pth")

    @classmethod
    def load(
            cls,
            path,
            env: Optional[GymEnv] = None,
            device: Union[th.device, str] = "auto",
            custom_objects: Optional[Dict[str, Any]] = None,
            force_reset: bool = True,
            **kwargs,
    ):

        data, params, pytorch_variables = load_from_zip_file(
            path, device=device, custom_objects=custom_objects, print_system_info=False
        )

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        tclf = th.load(f"{os.path.dirname(path)}/tclf.pth")
        # noinspection PyArgumentList
        model = cls(  # pytype: disable=not-instantiable,wrong-keyword-args
            tclf=tclf,
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, name + ".data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # pytype: disable=attribute-error

        return model
