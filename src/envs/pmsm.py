from typing import Optional

import gymnasium as gym
import numpy as np
import scipy.signal as signal
from gymnasium import spaces

from .utils import *


class EnvPMSM(gym.Env):
    def __init__(
        self,
        render_mode: Optional[str] = None,
        dt=1 / 10e3,
        r=29.0808e-3,
        ld=0.91e-3,
        lq=1.17e-3,
        lambda_PM=0.172312604,
        vdc=1200,
        we_nom=200 * 2 * np.pi,
        i_max=200,
        reward="absolute",
    ):
        # System parameters
        self.dt = dt  # Simulation step time [s]
        self.r = r  # Phase Stator Resistance [Ohm]
        self.ld = ld  # D-axis Inductance [H]
        self.lq = lq  # Q-axis Inductance [H]
        self.lambda_PM = lambda_PM  # Flux-linkage due to permanent magnets [Wb]
        self.we_nom = we_nom  # Nominal speed [rad/s]

        # Reward function type
        self.reward_function = reward

        # Maximum voltage [V]
        self.vdq_max = vdc / 2

        # Maximum current [A]
        self.i_max = i_max

        # Steady-state analysis functions
        self.ss_analysis = SSAnalysis()

        # Limitations for the system
        # Actions
        self.min_vd, self.max_vd = [-1.0, 1.0]
        self.min_vq, self.max_vq = [-1.0, 1.0]

        self.low_actions = np.array([self.min_vd, self.min_vq], dtype=np.float32)
        self.high_actions = np.array([self.max_vd, self.max_vq], dtype=np.float32)

        # Observations
        self.min_id, self.max_id = [-1.0, 1.0]
        self.min_iq, self.max_iq = [-1.0, 1.0]
        self.min_ref_id, self.max_ref_id = [-1.0, 1.0]
        self.min_ref_iq, self.max_ref_iq = [-1.0, 1.0]
        self.min_we, self.max_we = [-1.0, 1.0]
        self.min_vd, self.max_vd = [-1.0, 1.0]
        self.min_vq, self.max_vq = [-1.0, 1.0]

        self.low_observations = np.array(
            [
                self.min_id,
                self.min_iq,
                self.min_ref_id,
                self.min_ref_iq,
                self.min_we,
                self.min_vd,
                self.min_vq,
            ],
            dtype=np.float32,
        )
        self.high_observations = np.array(
            [
                self.max_id,
                self.max_iq,
                self.max_ref_id,
                self.max_ref_iq,
                self.max_we,
                self.max_vd,
                self.max_vq,
            ],
            dtype=np.float32,
        )

        # Render mode
        self.render_mode = render_mode

        # Define action and observation space within a Box property
        self.action_space = spaces.Box(low=self.low_actions, high=self.high_actions, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=self.low_observations,
            high=self.high_observations,
            shape=(7,),
            dtype=np.float32,
        )

    def step(self, action: np.ndarray):
        action_vdq = self.vdq_max * action  # Denormalize action

        # Calculate if that the module of Vdq is bigger than 1
        norm_vdq = np.sqrt(np.power(action_vdq[0], 2) + np.power(action_vdq[1], 2))
        # factor_vdq = self.vdq_max / norm_vdq
        # factor_vdq = factor_vdq if factor_vdq < 1 else 1
        factor_vdq = 1

        s_t = np.array([self.id, self.iq])
        a_t = factor_vdq * action_vdq

        # s(t+1) = ad * s(t) + bd * a(t) + w
        id_next, iq_next = self.ad @ s_t + self.bd @ a_t + self.wd
        # Rescale the current states to limit it within the boundaries if needed
        norm_idq_next = np.sqrt(np.power(id_next, 2) + np.power(iq_next, 2))
        factor_idq = self.i_max / norm_idq_next
        factor_idq = factor_idq if factor_idq < 1 else 1
        id_next, iq_next = factor_idq * np.array([id_next, iq_next])

        # Normalize observation
        id_next_norm = id_next / self.i_max
        iq_next_norm = iq_next / self.i_max
        id_ref_norm = self.id_ref / self.i_max
        iq_ref_norm = self.iq_ref / self.i_max
        we_norm = self.we / self.we_nom
        prev_vd_norm = self.prev_vd / self.vdq_max
        prev_vq_norm = self.prev_vq / self.vdq_max
        # Observation: [id, iq, id_ref, iq_ref, we, prev_vd, prev_vq]
        obs = np.array(
            [
                id_next_norm,
                iq_next_norm,
                id_ref_norm,
                iq_ref_norm,
                we_norm,
                prev_vd_norm,
                prev_vq_norm,
            ],
            dtype=np.float32,
        )

        terminated = False

        # Reward function
        id_norm = self.id / self.i_max
        iq_norm = self.iq / self.i_max
        e_id = np.abs(id_norm - id_ref_norm)
        e_iq = np.abs(iq_norm - iq_ref_norm)
        delta_vd = np.abs(action[0] - prev_vd_norm)
        delta_vq = np.abs(action[1] - prev_vq_norm)

        if self.reward_function == "absolute":
            reward = -(e_id + e_iq + 0.1 * (delta_vd + delta_vq))
        elif self.reward_function == "quadratic":
            reward = -((np.power(e_id, 2) + np.power(e_iq, 2)) + 0.1 * (np.power(delta_vd, 2) + np.power(delta_vq, 2)))
        elif self.reward_function == "quadratic_2":
            reward = -((np.power(e_id + e_iq, 2)) + 0.1 * (np.power(delta_vd + delta_vq, 2)))
        elif self.reward_function == "square_root":
            reward = -((np.power(e_id, 1 / 2) + np.power(e_iq, 1 / 2)) + 0.1 * (np.power(delta_vd, 1 / 2) + np.power(delta_vq, 1 / 2)))
        elif self.reward_function == "square_root_2":
            reward = -((np.power(e_id + e_iq, 1 / 2)) + 0.1 * (np.power(delta_vd + delta_vq, 1 / 2)))
        elif self.reward_function == "quartic_root":
            reward = -((np.power(e_id, 1 / 4) + np.power(e_iq, 1 / 4)) + 0.1 * (np.power(delta_vd, 1 / 4) + np.power(delta_vq, 1 / 4)))
        elif self.reward_function == "quartic_root_2":
            reward = -((np.power(e_id + e_iq, 1 / 4)) + 0.1 * (np.power(delta_vd + delta_vq, 1 / 4)))

        # Update states
        self.id = id_next
        self.iq = iq_next
        self.prev_vd = action_vdq[0]
        self.prev_vq = action_vdq[1]

        return obs, reward, terminated, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        low, high = 0.9 * np.array([-1, 1])
        # Initialization
        # [we]
        we_norm = np.round(self.np_random.uniform(low=0, high=high), 5)

        # Overwrite predefined speed from options
        if options:
            we_norm = np.float32(options.get("we_norm")) if options.get("we_norm") else we_norm

        # Define denormalized speed value
        we = we_norm * self.we_nom

        # we_norm = 0.1
        # [id,iq]
        id_norm = np.round(self.np_random.uniform(low=low, high=high), 5)
        iq_lim = np.sqrt(np.power(high, 2) - np.power(id_norm, 2))
        iq_norm = np.round(self.np_random.uniform(low=-iq_lim, high=iq_lim), 5)
        # [id_ref, iq_ref]
        id_ref_norm = np.round(self.np_random.uniform(low=low, high=high), 5)
        iq_ref_lim = np.sqrt(np.power(high, 2) - np.power(id_ref_norm, 2))
        iq_ref_norm = np.round(self.np_random.uniform(low=-iq_ref_lim, high=iq_ref_lim), 5)

        ## Testing points
        # we = 909.89321869 # [rad/s]
        # we_norm = 909.89321869/self.we_nom
        # id_norm = 0.1
        # iq_norm = -0.6
        # id_ref_norm = -0.6
        # iq_ref_norm = 0.33

        # dq-frame continuous state-space
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/Ld      we*Lq/Ld][Id]  +  [1/Ld      0 ][Vd] + [      0      ]
        # [dIq/dt]   [-we*Ld/Lq     -R/Lq][Iq]     [ 0      1/Lq][Vq]   [-we*lambda_PM]
        a = np.array(
            [
                [-self.r / self.ld, we * self.lq / self.ld],
                [-we * self.ld / self.lq, -self.r / self.lq],
            ]
        )
        b = np.array([[1 / self.ld, 0], [0, 1 / self.lq]])
        w = np.array([[0], [-we * self.lambda_PM]])
        c = np.eye(2)
        d = np.zeros((2, 2))

        bw = np.hstack((b, w))
        dw = np.hstack((d, np.zeros((2, 1))))
        (ad, bdw, _, _, _) = signal.cont2discrete((a, bw, c, dw), self.dt, method="zoh")

        # s_(t+1) = ad * s(t) + bd * a(t) + w
        # where ad and bd are 2x2 matrices, s(t) the state [Id, Iq], and a(t) the actions [Vd, Vq].
        # s(t) = dq currents
        # a(t) = dq voltages
        # w = disturbance due to flux-linkage from permanent magnets
        self.ad = ad
        self.bd = bdw[:, : b.shape[1]]
        self.wd = bdw[:, b.shape[1] :].squeeze()

        # Steady-state analysis
        # self.ss_analysis.continuous(a, b, w.squeeze(), plot_current=True)        # Continuous
        # self.ss_analysis.discrete(ad, self.bd, self.wd, plot_current=True)     # Discrete

        # Overwrite predefined current values from options
        if options:
            id_norm = np.float32(options.get("id_norm")) if options.get("id_norm") is not None else id_norm
            iq_norm = np.float32(options.get("iq_norm")) if options.get("iq_norm") is not None else iq_norm
            id_ref_norm = np.float32(options.get("id_ref_norm")) if options.get("id_ref_norm") is not None else id_ref_norm
            iq_ref_norm = np.float32(options.get("iq_ref_norm")) if options.get("iq_ref_norm") is not None else iq_ref_norm
            prev_vd_norm = np.float32(options.get("prev_vd_norm")) if options.get("prev_vd_norm") is not None else None
            prev_vq_norm = np.float32(options.get("prev_vq_norm")) if options.get("prev_vq_norm") is not None else None

            self.prev_vd = prev_vd_norm * self.vdq_max if prev_vd_norm is not None and prev_vq_norm is not None else None
            self.prev_vq = prev_vq_norm * self.vdq_max if prev_vd_norm is not None and prev_vq_norm is not None else None

            # Observation: [id, iq, id_ref, iq_ref, we, prev_vd, prev_vq]
            obs = np.array(
                [
                    id_norm,
                    iq_norm,
                    id_ref_norm,
                    iq_ref_norm,
                    we_norm,
                    prev_vd_norm,
                    prev_vq_norm,
                ],
                dtype=np.float32,
            )

        else:
            self.prev_vd = None
            self.prev_vq = None

        # Store idq, and idq_ref
        self.id = self.i_max * id_norm
        self.iq = self.i_max * iq_norm
        self.id_ref = self.i_max * id_ref_norm
        self.iq_ref = self.i_max * iq_ref_norm
        self.we = self.we_nom * we_norm

        # Additional steps to store previous actions
        if self.prev_vd is None or self.prev_vq is None:
            n = 2
            self.prev_vd = 0
            self.prev_vq = 0
            for _ in range(n):
                obs, _, _, _, _ = self.step(action=self.action_space.sample())
        return obs, {}
