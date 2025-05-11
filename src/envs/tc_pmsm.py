from typing import Optional

import gymnasium as gym
import numpy as np
import scipy.signal as signal
from gymnasium import spaces

from .utils import *
class EnvPMSMTC(gym.Env):
    def __init__(self, render_mode: Optional[str] = None,
                 dt = 1/10e3,
                 resistance = 29.0808e-3,
                 Ld = 0.91e-3,
                 Lq = 1.17e-3,
                 lambda_PM = 0.172312604,
                 Vdc = 1200,
                 we_norm = 200*2*np.pi,
                 I_max = 300,
                 te_max = 100,
                 reward = "absolute",
                 poles = 4
                 ):
        # System parameters
        self.dt     = dt      # Simulation step time [s]
        self.r      = resistance       # Phase Stator Resistance [Ohm]
        self.ld     = Ld     # D-axis Inductance [H]
        self.lq     = Lq      # Q-axis Inductance [H]
        self.lambda_PM = lambda_PM  # Flux-linkage due to permanent magnets [Wb]
        self.we_nom = we_norm  # Nominal speed [rad/s]
        vdc         = Vdc     # DC bus voltage [V]

        # Reward function type
        self.reward_function = reward

        # Maximum voltage [V]
        self.vdq_max = vdc/2

        # Maximum current [A]
        self.i_max  = I_max

        # Maximum torque [Nm]
        self.te_max  = te_max

        # Torque estimation
        self.psi_d = lambda id: self.ld*id + self.lambda_PM
        self.psi_q = lambda iq: self.lq*iq
        self.te_calculation = lambda id, iq: 3/2*poles*(self.psi_d(id)*iq - self.psi_q(iq)*id)

        # Steady-state analysis functions
        self.ss_analysis = SSAnalysis()

        # Limitations for the system
        # Actions
        self.min_vd, self.max_vd = [-1.0, 1.0]
        self.min_vq, self.max_vq = [-1.0, 1.0]

        self.low_actions = np.array(
            [self.min_vd, self.min_vq], dtype=np.float32
        )
        self.high_actions = np.array(
            [self.max_vd, self.max_vq], dtype=np.float32
        )

        # Observations
        self.min_te,     self.max_te     = [-1.0, 1.0]
        self.min_ref_te, self.max_ref_te = [-1.0, 1.0]
        self.min_id,     self.max_id     = [-1.0, 1.0]
        self.min_iq,     self.max_iq     = [-1.0, 1.0]
        self.min_we,     self.max_we     = [-1.0, 1.0]
        self.min_vd,     self.max_vd     = [-1.0, 1.0]
        self.min_vq,     self.max_vq     = [-1.0, 1.0]

        self.low_observations = np.array(
            [self.min_te, self.min_ref_te, self.min_id, self.min_iq, self.min_we, self.min_vd, self.min_vq], dtype=np.float32
        )
        self.high_observations = np.array(
            [self.max_te, self.max_ref_te, self.max_id, self.max_iq, self.max_we, self.max_vd, self.max_vq], dtype=np.float32
        )

        # Render mode
        self.render_mode = render_mode

        # Define action and observation space within a Box property
        self.action_space = spaces.Box(
            low=self.low_actions, high=self.high_actions, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_observations, high=self.high_observations, shape=(7,), dtype=np.float32
        )

    def step(self, action: np.ndarray):
        action_vdq = self.vdq_max * action  # Denormalize action

        # Calculate if that the module of Vdq is bigger than 1
        norm_vdq = np.sqrt(np.power(action_vdq[0], 2) + np.power(action_vdq[1], 2))
        # factor_vdq = self.vdq_max / norm_vdq
        # factor_vdq = factor_vdq if factor_vdq < 1 else 1
        factor_vdq = 1

        s_t = np.array([self.id,
                        self.iq])
        a_t = factor_vdq * action_vdq

        # s(t+1) = ad * s(t) + bd * a(t) + w
        id_next, iq_next = self.ad @ s_t + self.bd @ a_t + self.wd
        # Rescale the current states to limit it within the boundaries if needed
        norm_idq_next = np.sqrt(np.power(id_next, 2) + np.power(iq_next, 2))
        factor_idq = self.i_max / norm_idq_next
        factor_idq = factor_idq if factor_idq < 1 else 1
        id_next, iq_next = factor_idq * np.array([id_next, iq_next])

        # Estimate new electric torque
        te_next = self.te_calculation(id_next, iq_next)

        # Normalize observation
        te_next_norm = te_next / self.te_max
        te_ref_norm  = self.te_ref / self.te_max
        id_next_norm = id_next / self.i_max
        iq_next_norm = iq_next / self.i_max
        we_norm      = self.we / self.we_nom
        prev_vd_norm = self.prev_vd / self.vdq_max
        prev_vq_norm = self.prev_vq / self.vdq_max
        # Observation: [te, te_ref, id, iq, we, prev_vd, prev_vq]
        obs = np.array([te_next_norm, te_ref_norm, id_next_norm, iq_next_norm,  we_norm, prev_vd_norm, prev_vq_norm], dtype=np.float32)

        terminated = False

        # Reward function
        te_norm = self.te_calculation(self.id, self.iq) / self.te_max
        i_dq_mag = np.sqrt(np.power(self.id, 2) + np.power(self.iq, 2)) / self.i_max
        e_te = np.abs(te_norm - te_ref_norm)
        delta_vd = np.abs(action[0] - prev_vd_norm)
        delta_vq = np.abs(action[1] - prev_vq_norm)

        # Scaling factor for objectives
        w_idq = 0.1
        w_vdq = 0.1
        if self.reward_function == "absolute":
            reward = -(e_te + w_idq * i_dq_mag + w_vdq * (delta_vd + delta_vq))
        elif self.reward_function == "quadratic":
            reward = -(np.power(e_te, 2) + w_idq * np.power(i_dq_mag, 2) + w_vdq * (np.power(delta_vd, 2) + np.power(delta_vq, 2)))
        elif self.reward_function == "quadratic_2":
            reward = -(np.power(e_te, 2) + w_idq * np.power(i_dq_mag, 2) + w_vdq * (np.power(delta_vd + delta_vq, 2)))
        elif self.reward_function == "square_root":
            reward = -(np.power(e_te, 1/2) + w_idq * np.power(i_dq_mag, 1/2) + w_vdq * (np.power(delta_vd, 1/2) + np.power(delta_vq, 1/2)))
        elif self.reward_function == "square_root_2":
            reward = -(np.power(e_te, 1/2) + w_idq * np.power(i_dq_mag, 1/2) + w_vdq * (np.power(delta_vd + delta_vq, 1/2)))
        elif self.reward_function == "quartic_root":
            reward = -(np.power(e_te, 1/4) + w_idq * np.power(i_dq_mag, 1/4) + w_vdq * (np.power(delta_vd, 1/4) + np.power(delta_vq, 1/4)))
        elif self.reward_function == "quartic_root_2":
            reward = -(np.power(e_te, 1/4) + w_idq * np.power(i_dq_mag, 1/4) + w_vdq * (np.power(delta_vd + delta_vq, 1/4)))

        # Update states
        self.id = id_next
        self.iq = iq_next
        self.prev_vd = action_vdq[0]
        self.prev_vq = action_vdq[1]

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        # Boundary for initialization values
        low, high = 0.9 * np.array([-1, 1])

        # Initialization speed
        # [we]
        we_norm = self.np_random.uniform(low=0, high=high)

        # Overwrite predefined speed from options
        if options:
            we_norm = np.float32(options.get("we_norm")) if options.get("we_norm") else we_norm

        # Define denormalized speed value
        we = we_norm * self.we_nom

        # dq-frame continuous state-space
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/Ld      we*Lq/Ld][Id]  +  [1/Ld      0 ][Vd] + [      0      ]
        # [dIq/dt]   [-we*Ld/Lq     -R/Lq][Iq]     [ 0      1/Lq][Vq]   [-we*lambda_PM]
        a = np.array([[-self.r / self.ld,           we * self.lq / self.ld],
                      [-we * self.ld / self.lq,     -self.r / self.lq]])
        b = np.array([[1 / self.ld, 0],
                      [0, 1 / self.lq]])
        w = np.array([[0], [-we * self.lambda_PM]])
        c = np.eye(2)
        d = np.zeros((2,2))

        bw = np.hstack((b, w))
        dw = np.hstack((d, np.zeros((2,1))))
        (ad, bdw, _, _, _) = signal.cont2discrete((a, bw, c, dw), self.dt, method='zoh')

        # s_(t+1) = ad * s(t) + bd * a(t) + w
        # where ad and bd are 2x2 matrices, s(t) the state [Id, Iq], and a(t) the actions [Vd, Vq].
        # s(t) = dq currents
        # a(t) = dq voltages
        # w = disturbance due to flux-linkage from permanent magnets
        self.ad = ad
        self.bd = bdw[:,:b.shape[1]]
        self.wd = bdw[:,b.shape[1]:].squeeze()

        # Steady-state analysis
        # self.ss_analysis.continuous(a, b, w.squeeze(), plot_current=True)        # Continuous
        # self.ss_analysis.discrete(ad, self.bd, self.wd, plot_current=True)     # Discrete

        # Initialization currents
        # [id,iq]
        id_norm = self.np_random.uniform(low=low, high=high)
        iq_lim  = np.sqrt(np.power(high,2)  - np.power(id_norm,2))
        iq_norm = self.np_random.uniform(low=-iq_lim, high=iq_lim)
        # [te_ref]
        te_ref_norm = self.np_random.uniform(low=low, high=high)

        # Overwrite predefined current values from options
        if options:
            id_norm      = np.float32(options.get("id_norm"))      if options.get("id_norm")      is not None else id_norm
            iq_norm      = np.float32(options.get("iq_norm"))      if options.get("iq_norm")      is not None else iq_norm
            te_ref_norm  = np.float32(options.get("te_ref_norm"))  if options.get("te_ref_norm")  is not None else te_ref_norm
            prev_vd_norm = np.float32(options.get("prev_vd_norm")) if options.get("prev_vd_norm") is not None else None
            prev_vq_norm = np.float32(options.get("prev_vq_norm")) if options.get("prev_vq_norm") is not None else None

            self.prev_vd = prev_vd_norm * self.vdq_max if prev_vd_norm is not None and prev_vq_norm is not None else None
            self.prev_vq = prev_vq_norm * self.vdq_max if prev_vd_norm is not None and prev_vq_norm is not None else None
            
            # Calculate Te
            te_norm = self.te_calculation(self.id, self.iq) / self.te_max
            # Observation: [te, te_ref, id, iq, we, prev_vd, prev_vq]
            obs = np.array([te_norm, te_ref_norm, id_norm, iq_norm,  we_norm, prev_vd_norm, prev_vq_norm], dtype=np.float32)
        else:
            self.prev_vd = None
            self.prev_vq = None

        
        # Store idq, and idq_ref
        self.id     = self.i_max * id_norm
        self.iq     = self.i_max * iq_norm
        self.te_ref = self.te_max * te_ref_norm
        self.we     = self.we_nom * we_norm

        # Additional steps to store previous actions
        if self.prev_vd is None or self.prev_vq is None:
            n = 2
            self.prev_vd = 0
            self.prev_vq = 0
            for _ in range(n):
                obs, _, _, _, _ = self.step(action=self.action_space.sample())
        return obs, {}