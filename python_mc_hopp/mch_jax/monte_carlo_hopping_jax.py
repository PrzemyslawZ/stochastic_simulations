from jax import jit, lax, random, clear_caches, config, checking_leaks
from jax.lib import xla_bridge
from functools import partial
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import interp1d

import os

# 1) TODO: Check if simulation gives correct results (DEBUGG) - DONE
# 2) TODO: Compare results with with scipy vs loop with numpy interpolation  - BETTER interpp1d
# 3) TODO: Compare jax (after choosing better option in 2) with clean python)
# 4) TODO: Implement multiprocessing and add common interface !

class MCHopping_jax:

	def __init__(self, params:dict={}, device=False):
		config.update('jax_platform_name', 'gpu' if device else 'cpu')
		self.params = params
		self._check()
		self._initialize()
		os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={self.params["num_cpu"]}'
		print(f"[INFO] Device: {xla_bridge.get_backend().platform}")
		
	def simulate(self):

		self.avg_population = []
		for gain_r, loss in zip(self.params["gains_right"], self.params["losses_r"]):
			for hop_rate in self.params["hop_rates"]: 
	
				self._loss_r = loss
				self._gain_r = gain_r  
				self._hop_rate = hop_rate
				# self.mp_handler.run_workers(self._mcarlo_hopping)
				result = self._mcarlo_hopping(self.params["trajectories_num"])
				# self.avg_population.append(
				# 	np.sum(self.mhopp.result, axis=0)/self.params["trajectories_num"])
				self.avg_population.append(result/ self.params["trajectories_num"])
		self.results = self.avg_population

	def _mcarlo_hopping(self, work_load:int):
		for i in range(work_load):
			n_store, times, idx = self._mcarlo_hopping_wrapper(self._keys[i])
			self._adjust(n_store, times, idx)
			self.n_per_worker += self.population
		return self.n_per_worker
		
	@partial(jit, static_argnums=(0,))
	def _mcarlo_hopping_wrapper(self, key):
		idx, t, _, _, times, n_ch, n_store = lax.while_loop(
			cond_fun=self._condition, 
			body_fun=self._run_trajectory, 
			init_val=(
				self.idx, self.t, self.events, key,
				self.times, self.n_ch, self.n_store)
				)
		times = times.at[idx].set(t + 2 * 1e-8)
		return n_store, times, idx+1

	# @partial(jit, static_argnums=(0,))
	# def _complete(self, data, times):
	# 	_, population, _, _ = lax.fori_loop(
	# 		0, data.shape[0], 
	# 		body_fun=self._interpolate, 
	# 		init_val=(data, self.population, self.params["time"], times), 
	# 		unroll=True
	# 		)
	# 	return population.T

	def _adjust(self, n_store, times, idx):
		n_store = lax.dynamic_slice_in_dim(n_store, 0, idx, axis=-1)
		times = lax.dynamic_slice_in_dim(times, 0, idx, axis=0)
		self._interpolate(n_store, times)
		return n_store, times

	def _interpolate(self, n_store, times):
		'''
		Method interpolates stored excitations on finer time grid 
		
		Args: 
			None

		Return:
			None
		
		'''
		
		self.population = interp1d(np.asarray(times), n_store.T, axis=0)(self.params["time"])

	# @partial(jit, static_argnums=(0,))
	def _run_trajectory(self, args):
		idx, t, events, key_init, times, n_ch, n_store = args
		events = events.at[:self.N-1].set(self._hop_rate * n_ch[1:] * (1 + n_ch[:self.N-1]))
		events = events.at[self.N-1].set(self._loss_r * n_ch[-1])
		events = events.at[self.N].set(self._gain_r * (1 + n_ch[-1]))
		events = events.at[self.N+1].set(self.params["loss_l"] * n_ch[0])
		events = events.at[self.N+2:].set(self.params["loss_chain"] * n_ch)

		key, subkey_1, subkey_2 = random.split(key_init, num=3)
		del key_init
		norm = jnp.sum(events)
		
		t, prob_sel = lax.cond(
			norm!=0,
			self._hopp, 
			self._no_hopp, t, events, norm, subkey_1, subkey_2)
		del subkey_1, subkey_2
		
		times=times.at[idx+1].set(t - 1e-8)
		times=times.at[idx+2].set(t + 1e-8)

		n_ch, n_store = self._update(idx, n_store, n_ch, prob_sel)
		return idx+1, t, events, key, times, n_ch, n_store
		
	# @partial(jit, static_argnums=(0,))
	def _condition(self, args):
		return (args[1] <= self.t_end)

	# @partial(jit, static_argnums=(0,))
	def _hopp(self, t, events, norm, key1, key2):
		dt = -jnp.log(random.uniform(key1)) / norm
		prob_sel = random.binomial(key1, 1, events/norm)
		return t+dt, prob_sel
		
	# @partial(jit, static_argnums=(0,))
	def _no_hopp(self, t, events, norm, key1, key2):
		prob_sel = jnp.zeros(events.size)
		return jnp.float32(t + self.params["t_cut"]), prob_sel

	# @partial(jit, static_argnums=(0,))
	def _update(self, idx, n_store, n_ch, prob_sel):
		n_ch = n_ch.at[:].add(jnp.append(prob_sel[0:self.N-1], jnp.array([0])) -\
		jnp.append(jnp.array([0]), prob_sel[0:self.N-1]) - prob_sel[self.N+2:])
		n_ch = n_ch.at[0].add(-prob_sel[self.N+1])
		n_ch = n_ch.at[-1].add((-prob_sel[self.N-1] + prob_sel[self.N]))

		n_store = n_store.at[:,idx+1].set(n_ch)
		return n_ch, n_store

	def _check(self):
		'''
		Method checks validity of the parameters. In 
		case of empty params dict standard set of settings 
		will be provided

		Args: 
			None

		Return:
			None
		
		'''
		
		if self.params == {}:
			self.params = {
				"N": 10,
				"trajectories_num": 10,
				"hop_rates": [5],
				"gains_right": [5],
				"losses_r": [0],
				"loss_l": 1,
				"loss_chain": 0,
				"time": jnp.linspace(0, 300, int(300/1e-3)),
				"t_cut": 10,
				"dt": 1e-8,
				"num_cpu": None,
			}
		else: pass
	
	def _initialize(self):
		'''
		Method initializes all parameters and variables needed
		during Monte Carlo simulation

		Args: 
			None

		Return:
			None
		
		'''
		
		self.t = 0
		self.idx = 0
		self.N = self.params["N"]
		self.n_ch = jnp.zeros(self.N)
		self.events = jnp.zeros(2*self.N+2)
		self.t_end = self.params["time"][-1]
		self.times = jnp.zeros(self.params["time"].shape)
		self.n_per_worker = np.zeros((self.params["time"].size, self.N))
		self.n_store = jnp.zeros((self.N, self.params["time"].size))
		self._keys = [random.key(seed) for seed in np.random.randint(0,100, size=self.params["trajectories_num"])]