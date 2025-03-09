from mch import MCHopping as mc
from mch import np, plt
from mch_jax import MCHopping_jax as mc_jax

class MCHandler:
	'''
	Handler for MonteCarlo simualtions 

	'''
	def __init__(self, params:dict={}, mch_jax:bool=False, device=False):
		
		'''
		Contructor

		Args: 
			params: dict=None - dictionary with simulation parameters
			mch_jax: bool=False - param controlling type of code (jax/python)
			device: bool=False - flag for using GPU device

		Return:
			None
		
		'''

		self.params = params
		self._check()
		
		self.mc_hopp = mc_jax(self.params, device) if mch_jax else mc(self.params)

	def run(self)->None:
		'''
		Method calls Monte Carlo simulation for either clear Python or JAX 
		'''
		self.mc_hopp.simulate()

	def __del__(self):
		'''
		Destructor

		Args: 
			None

		Return:
			None
		'''

		del self.mc_hopp, self.params

	
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
			data = self.mc_hopp.results[0]
		else:
			pass
		
		plt.figure(figsize=(8,3))
		plt.subplot(121)
		plt.title("Site occupation at final time")
		plt.plot(np.arange(1,self.params["N"]+1, 1), data[-1, :])
		plt.xlabel("n site")
		plt.ylabel("Population")
		plt.subplot(122)
		plt.title("Evoluon of the chain occupation")
		plt.plot(self.params["time"], data[:, -1])
		plt.xlabel("t")
		plt.ylabel("Population")
		plt.savefig(f"./{file_name}.png")
		plt.close()
	
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




