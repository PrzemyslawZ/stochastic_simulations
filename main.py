from mch_controller import MCHandler
import datetime 
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt


def get_mc_simulation(N, n_traject, t_end, t_steps, mch_jax=True, device=False):
	
	params = {
				"N": N,
				"trajectories_num": n_traject,
				"hop_rates": [5],
				"gains_right": [5],
				"losses_r": [0],
				"loss_l": 1,
				"loss_chain": 0,
				"time": np.linspace(0, t_end, t_steps),
				"t_cut": 0.5,
				"dt": 1e-8,
				"num_cpu": 16
			}

	mch_handl = MCHandler(params, mch_jax=mch_jax, device=device)

	start = datetime.datetime.now()
	mch_handl.run()
	end = datetime.datetime.now()

	elpst = (end-start).seconds
	print(f"Elapsed time: {elpst}")

	mch_handl.plot(mch_handl.mc_hopp.results[0], 
	file_name=f"N={N}_n_traj={tr}+jax_avg_population")
	
	del mch_handl

	return elpst


def jax_perf_test():
	data = {}
	traj = [2, 4, 16, 32, 64, 128, 256, 512, 1024, 2048]
	N = [5, 10, 15, 20]
	for n in N:
		data[n] = []
		for tr in traj:
			data[n].append(get_mc_simulation(n, tr, 300, 70000))

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


def main():

	params = {
			"N": 10,
			"trajectories_num": 10,
			"hop_rates": [5],
			"gains_right": [5],
			"losses_r": [0],
			"loss_l": 1,
			"loss_chain": 0,
			"time": np.linspace(0, 300, 40000),
			"t_cut": 0.5,
			"dt": 1e-8,
			"num_cpu": 5
			}

	args_num = len(sys.argv)

	mch_jax = True if (args_num > 1) and sys.argv[1] == "jax" else False
	jax_device = True if (args_num > 2) and sys.argv[2] == "gpu" else False

	mch_handl = MCHandler(params, mch_jax=mch_jax, device=jax_device)

	start = datetime.datetime.now()
	mch_handl.run()
	end = datetime.datetime.now()
	
	elpst = (end-start)
	print(f"Elapsed time: {elpst}")
	mch_handl.plot(mch_handl.mc_hopp.results[0], 
	file_name=f"N={params["N"]}_n_traj={params["trajectories_num"]}+jax={mch_jax}_avg_population")

	del mch_handl

if __name__ == '__main__':
	main()