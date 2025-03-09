from mch_controller import MCHandler
import datetime 
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
# @jit
# def f(x):
# 	return x*2


# def g(x):
# 	return f(x)

# 	with mp.Pool(processes=4) as pool:
# 		result = pool.map(g, [1,1,1,1])
# 	print(result)

def main():

	data = {}
	traj = [2, 4, 16, 32, 64, 128, 256, 512, 1024, 2048]
	N = [5, 10, 15, 20]
	for n in N:
		data[n] = []
		for tr in traj:
			params = {
					"N": n,
					"trajectories_num": tr,
					"hop_rates": [5],
					"gains_right": [5],
					"losses_r": [0],
					"loss_l": 1,
					"loss_chain": 0,
					"time": np.linspace(0, 300, 70000),
					"t_cut": 0.5,
					"dt": 1e-8,
					"num_cpu": 16
					}
		
			# args_num = len(sys.argv)
			
			mch_jax = True #if (args_num > 1) and sys.argv[1] == "jax" else False
			jax_device = False #True if (args_num > 2) and sys.argv[2] == "gpu" else False
		
			mch_handl = MCHandler(params, mch_jax=mch_jax, device=jax_device)
		
			start = datetime.datetime.now()
			mch_handl.run()
			end = datetime.datetime.now()
			
			elpst = (end-start).seconds
			print(f"Elapsed time: {elpst}")
			data[n].append(elpst)
			mch_handl.plot(mch_handl.mc_hopp.results[0], file_name=f"N={N}_n_traj={tr}+jax_avg_population")
			del mch_handl

	plt.figure(figsize=(6,3))
	plt.title(f"t_end={params["time"][-1]}_tot_tsteps={params["time"].size}")
	for n in N:
		plt.plot(traj, data[n], label=f"N={n}")
		plt.xlabel("# trajectories")
		plt.ylabel("times [s]")
		plt.legend()
	plt.savefig("all_perf_jax_test.png")
	
	with open('perf_jax_data.pkl', 'wb') as f:
	    pickle.dump(data, f)
	# del mch_handl

if __name__ == '__main__':
	main()