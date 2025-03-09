import multiprocessing as mp

class MultiHopp:

	'''
	Interface for multiprocessing Monte Carlo simulation
	
	'''
	
	def __init__(self, num_cpu:int, total_iterations:int, init_func:callable=None):
		'''
		Contructor

		Args: 
			num_cpu: int - number of processes to run in parallel
			total_iterations: int - number of trajectories performed 
			by single thread
			init_func: callable - method executed before all workers

		Return:
			None
			
		'''
		
		self._count(num_cpu)
		self._initializer = init_func
		
		self._tot_iter = total_iterations
		self._load = self._tot_iter // self.ncpu
	
	def run_workers(self, func):
		'''
		Method executes parallel computation defined within func.

		Args: 
			func: Callable - method to be executed in parallel

		Return:
			None
		
		'''
		
		if self._load !=0 :
			work_load = [self._tot_iter // self.ncpu] * self.ncpu
			print(f"[INFO] {self.ncpu} workers starting job with {self._load} load each")
			with mp.Pool(processes=self.ncpu, initializer=self._initializer) as pool:
				self.result = pool.map(func, work_load)
		else:
			print(f"[ERROR] Not enough trajectories, set at least {self.ncpu}")
	
	def _count(self, num_cpu:int=None):
		'''
		Method checks and sets number of available processes to 
		perform computation

		Args: 
			num_cpu: int=None - number of processes

		Return:
			None
		
		'''
		try:
			self.ncpu = num_cpu if num_cpu is not None else mp.cpu_count()
		except:
			self.ncpu = 1
