import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from parallelization import MultiHopp, mp
	
	
class MCHopping:
	'''
	Interface for MonteCarlo simulation of asymertic bosonic transport
	
	'''
	
	def __init__(self, params:dict={}):
		'''
		Contructor

		Args: 
			params: dict=None - dictionary with simulation parameters

		Return:
			None
		
		'''
		
		self.params = params
		self._check()
		self._initialize()
		self.mp_handler = MultiHopp(
			self.params["num_cpu"], 
			self.params["trajectories_num"], 
			self._seed)
		
	
	def __del__(self):
		'''
		Destructor

		Args: 
			params: dict - dictionary with simulation parameters

		Return:
			None
		
		'''
		
		del self.n_ch, self.mp_handler
		del self.events, self.n_per_worker
		del self.times, self.params

	def simulate(self):
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
				self.mp_handler.run_workers(self._mcarlo_hopping)
				self.avg_population.append(
					np.sum(self.mp_handler.result, axis=0)/self.params["trajectories_num"])
		self.results = self.avg_population
		
	def _mcarlo_hopping(self, work_load:int):
		'''
		Method performs Monte Carlo simulation over assumed number
		of the trajectories. Execution takes place in parrallel

		Args: 
			work_load: int - number of trajectories performed by 
			single process

		Return:
			sum of the excitations in the chain over performed trajectories 
		
		'''
		
		for _ in range(work_load):
			self._run_trajectory()
			self.n_per_worker += self.population
			self._restart()
		print(f"[INFO] Worker {mp.current_process()._identity[0]} finished job")
		return self.n_per_worker
	
	def plot(self, data:(list, np.array)=None, file_name:str="avg_population"):
		'''
		Method plots and saves result of the simulation. In case data arg is not 
		provided first result of avg_population list will be displayed

		Args: 
			data: (list, np.array)=None- data to plot
			file_name: str=avg_population - name of saving png file

		Return:
			None
		
		'''
		
		if data is None:
			data = self.avg_population[0]
		else:
			pass
		
		plt.figure(figsize=(8,3))
		plt.subplot(121)
		plt.title("Site occupation at final time")
		plt.plot(np.arange(1,self.N+1, 1), data[-1, :])
		plt.xlabel("n site")
		plt.ylabel("Population")
		plt.subplot(122)
		plt.title("Evoluon of the chain occupation")
		plt.plot(self.params["time"], data[:, -1])
		plt.xlabel("t")
		plt.ylabel("Population")
		plt.savefig(f"./{file_name}.png")
		plt.close()
	
	def _run_trajectory(self):
		'''
		Method performs single Monte Carlo trajectory simulation 

		Args: 
			None

		Return:
			None
		
		'''
		i = 0
		while self.t <= self.t_end:
			self.events[:self.N-1] = self._hop_rate * self.n_ch[1:] * (1 + self.n_ch[:self.N-1])
			self.events[self.N-1] = self._loss_r * self.n_ch[-1]
			self.events[self.N] =  self._gain_r * (1 + self.n_ch[-1])
			self.events[self.N+1] = self.params["loss_l"] * self.n_ch[0]
			self.events[self.N+2:] = self.params["loss_chain"] * self.n_ch

			random_event = np.random.uniform()
			prob_sum = np.sum(self.events)
			if prob_sum != 0:
				self.t += -np.log(random_event) / prob_sum
				self.prob_sel = np.random.multinomial(1, self.events / prob_sum)
			else:
				self.t += self.params["t_cut"]
				self.prob_sel = np.zeros(self.events.size)
			
			self._update()
			self.times.extend([self.t - self.params["dt"], self.t + self.params["dt"]])
			i+=1
			
		self.times.append(self.t + 2 * self.params["dt"])
		self._interpolate()

	def _interpolate(self):
		'''
		Method interpolates stored excitations on finer time grid 
		
		Args: 
			None

		Return:
			None
		
		'''
		self.population = interp1d(
			np.asarray(self.times), 
			self.n_store.T, axis=0)(self.params["time"])
		
	def _update(self):
		'''
		Method updates number of excitations for single time steps and
		updates system evoluion of the chain

		Args: 
			None

		Return:
			None
		
		'''
		
		self.n_ch += np.append(self.prob_sel[0:self.N-1], [0]) -\
		np.append([0], self.prob_sel[0:self.N-1]) - self.prob_sel[self.N+2:]
		self.n_ch[0] -=  self.prob_sel[self.N+1]
		self.n_ch[-1] += (-self.prob_sel[self.N-1] + self.prob_sel[self.N])
		self.n_store = np.column_stack([self.n_store, self.n_ch, self.n_ch])  
		
	@staticmethod
	def _seed():
		'''
		Method initialize seed for each process

		Args: 
			None

		Return:
			None
		
		'''
		
		np.random.seed() 
		
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
				"time": np.linspace(0, 300, int(300/1e-3)),
				"t_cut": 10,
				"dt": 1e-8,
				"num_cpu": None,
			}
		else: pass
	
	def _restart(self):
		'''
		Method resets system for trajectory simulation

		Args: 
			None

		Return:
			None
		
		'''
		
		self.times = [0]
		self.n_ch *= 0
		self.n_store = np.column_stack([self.n_ch, self.n_ch])
		self.t = 0
		
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
		self.N = self.params["N"]
		self.t_end = self.params["time"][-1]
		self.times = [0]
		self.n_per_worker = np.zeros((self.params["time"].size, self.N))
	
		self.n_ch = np.zeros(self.N)
		self.events = np.zeros(2*self.N+2)
		self.n_store = np.column_stack([self.n_ch, self.n_ch])
	
