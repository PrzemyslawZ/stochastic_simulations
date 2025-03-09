from jax import jit, lax, random, config
from jax.lib import xla_bridge
from functools import partial
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import interp1d
import os

class MCHopping_jax:
	'''
	Interface for MonteCarlo simulation of asymertic bosonic transport - JAX version
	
	'''
	def __init__(self, params:dict={}, device:bool=False):
		'''
		Contructor

		Args: 
			params: dict=None - dictionary with simulation parameters
			device: bool=False - flag for using GPU device

		Return:
			None
		
		'''
		
		config.update('jax_platform_name', 'gpu' if device else 'cpu')
		print(f"[INFO] Device: {xla_bridge.get_backend().platform}")
		self.params = params
		self._initialize()

	def __del__(self):
		'''
		Destructor

		Args: 
			None

		Return:
			None
		
		'''

		del self.n_ch, self.n_per_worker
		del self.events, self.times
		del self.params, self._keys
		
	def simulate(self)->None:
		'''
		Method executes physical simulation with nested loops over
		different gain, hop and loss rates. 

		Args: 
			None

		Return:
			None

		'''

		self.avg_population = []
		for gain_r, loss in zip(self.params["gains_right"], self.params["losses_r"]):
			for hop_rate in self.params["hop_rates"]: 
	
				self._loss_r = loss
				self._gain_r = gain_r  
				self._hop_rate = hop_rate
				result = self._mcarlo_hopping(self.params["trajectories_num"])
				self.avg_population.append(result/ self.params["trajectories_num"])

		self.results = self.avg_population

	def _mcarlo_hopping(self, work_load:int)->jnp.array:
		'''
		Method performs Monte Carlo simulation over assumed number
		of the trajectories.

		Args: 
			work_load: int - number of trajectories performed during simulation

		Return:
			self.n_per_workers - sum of the excitations in the chain over performed trajectories 
		
		'''
		for i in range(work_load):
			n_store, times, idx = self._mcarlo_hopping_wrapper(self._keys[i])
			self._adjust(n_store, times, idx)
			self.n_per_worker += self.population
		return self.n_per_worker
		
	@partial(jit, static_argnums=(0,))
	def _mcarlo_hopping_wrapper(self, key:jnp.array)->(jnp.array,jnp.array, int):
		'''
		Method wraps JAX while_loop of Monte Carlos trajectory simulation 
		to create interface for non-traced oprations
		
		Args: 
			key: jnp.array - pseudo-random number generator

		Return:
			(n_store, times, idx+1): tuple - trajectory of the chain, evoluiton time steps 
			and number of iterations
		
		'''

		idx, t, _, _, times, n_ch, n_store = lax.while_loop(
			cond_fun=self._condition, 
			body_fun=self._run_trajectory, 
			init_val=(
				self.idx, self.t, self.events, key,
				self.times, self.n_ch, self.n_store)
				)
		times = times.at[idx].set(t + 2 * 1e-8)
		return n_store, times, idx+1

	def _adjust(self, n_store:jnp.array, times:jnp.array, idx:int)->(jnp.array,jnp.array):
		'''
		Method drops zeros of buffers storing data 
		
		Args: 
			n_store: jnp.array - buffer with trajectory of the chain
			times: jnp.array - buffer with evolution time steps
			idx: int - index of the last valid argument in the buffer

		Return:
			(n_store, times): tuple - trajectory for chain, evoluiton time steps 
			without unnecessary zeros
		
		'''

		n_store = lax.dynamic_slice_in_dim(n_store, 0, idx, axis=-1)
		times = lax.dynamic_slice_in_dim(times, 0, idx, axis=0)
		self._interpolate(n_store, times)
		return n_store, times

	def _interpolate(self, n_store, times)->None:
		'''
		Method interpolates stored excitations on finer time grid 
		
		Args: 
			None

		Return:
			None
		
		'''
		
		self.population = interp1d(np.asarray(times), n_store.T, axis=0)(self.params["time"])

	def _run_trajectory(self, args)->tuple:
		'''
		Method performs single Monte Carlo trajectory simulation 

		Args: 
			args - tuple with data for simulation 
			(events, key_init, times, n_ch, n_store) and total interation 
			trackers (idx, t)

		Return:
			tuple with: idx, t, events, key_init, times, n_ch, n_store 
		
		'''
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
		
	def _condition(self, args)->bool:
		'''
		Method checks condition (t <=t_end) for JAX while_loop 

		Args: 
			args - tuple with data for simulation 
			(events, key_init, times, n_ch, n_store) and total interation 
			trackers (idx, t)

		Return:
			condition: bool

		'''

		return (args[1] <= self.t_end)

	def _hopp(self, t:float, events:jnp.array, norm:int, key1:int, key2:int)->(float,jnp.array):
		'''
		Method executed in case of events buffer sum is different from zero;
		random jumps are performed

		Args: 
			t: float - current time
			events: jnp.array - buffer with simulated chain hopp events
			norm: int - events buffer sum
			key1: int - PRNG key
			key2: int - PRNG key

		Return:
			t+dt: float - incremented time
			prob_sel: jnp.array - buffer with random jumps 
			
		'''

		dt = -jnp.log(random.uniform(key1)) / norm
		prob_sel = random.binomial(key2, 1, events/norm)
		return t+dt, prob_sel

	def _no_hopp(self, t:float, events:jnp.array, norm:int, key1:int, key2:int)->(float,jnp.array):
		'''
		Method executed in case of events buffer sum is equal to zero;
		no random jumps are performed

		Args: 
			t: float - current time
			events: jnp.array - buffer with simulated chain hopp events
			norm: int - events buffer sum
			key1: int - PRNG key
			key2: int - PRNG key

		Return:
			t + self.params["t_cut"]: float - incremented time byt chosen cutoff time
			prob_sel: jnp.array - buffer with random jumps  (zeros)
			
		'''
		prob_sel = jnp.zeros(events.size)
		return jnp.float32(t + self.params["t_cut"]), prob_sel

	def _update(self, idx:int, n_store:jnp.array, n_ch:jnp.array, prob_sel:jnp.array)->(jnp.array,jnp.array):
		'''
		Method updates number of excitations for single time steps and
		updates system evoluion of the chain

		Args: 
			idx: int - current while loopp iteration
			n_store: jnp.array - buffer with chain trajectory history
			n_ch: jnp.array - buffer of current chain state
			prob_sel: jnp.array - buffer of random jumps

		Return:
			n_ch - buffer of updated chain state
			n_store - buffer with chain trajectory history
		
		'''

		n_ch = n_ch.at[:].add(jnp.append(prob_sel[0:self.N-1], jnp.array([0])) -\
		jnp.append(jnp.array([0]), prob_sel[0:self.N-1]) - prob_sel[self.N+2:])
		n_ch = n_ch.at[0].add(-prob_sel[self.N+1])
		n_ch = n_ch.at[-1].add((-prob_sel[self.N-1] + prob_sel[self.N]))

		n_store = n_store.at[:,idx+1].set(n_ch)
		return n_ch, n_store
	
	def _initialize(self)->None:
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