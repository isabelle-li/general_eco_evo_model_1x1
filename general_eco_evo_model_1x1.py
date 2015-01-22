from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp
from scipy.integrate import odeint
import timeit
import gc
import os
import json
#from time import sleep
import datetime

AVG_TIME_PER_GRAPH = 3.932
DIRECTORY = "/Users/samuelfleischer/Documents/python/general_eco_evo_model/1x1"
JSON = """{
	"system_parameters":[
		{	
			"steps": 30,

			"final_time" : 10000,

			"predator":{
				"initial_values":{
					"densities":{
						"1": {
							"start" : 100.0,
							"stop" : 100.0
						}
					},
					"traits":{
						"1": {
							"start": 0.0,
							"stop": 0.0
						}
					}
				},
				"trait_variances":{
					"total":{
						"1": {
							"start": 0.4,
							"stop": 0.4
						}
					},
					"genetic":{
						"1": {
							"start": 0.0,
							"stop": 0.6
						}
					}
				},
				"death_rates":{
					"1": {
						"start": 0.01,
						"stop": 0.01
					}
				}
			},
			"prey":{
				"initial_values":{
					"densities":{
						"1": {
							"start": 120.0,
							"stop": 120.0
						}
					},
					"traits":{
						"1": {
							"start": 0.1,
							"stop": 0.1
						}
					}
				},
				"trait_variances":{
					"total":{
						"1": {
							"start": 0.15,
							"stop": 0.15
						}
					},
					"genetic":{
						"1": {
							"start": 0.07,
							"stop": 0.07
						}
					}
				},
				"growth_rates":{
					"1": {
						"start": 0.1,
						"stop": 0.1
					}
				},
				"carrying_capacities":{
					"1": {
						"start": 225.0,
						"stop": 225.0
					}
				}
			},
			"interaction_parameters":{
				"efficiencies":{
					"11": {
						"start": 0.5,
						"stop": 0.5
					}
				},
				"specialization":{
					"11": {
						"start": 0.1,
						"stop": 0.1
					}
				},
				"max_attack_rates":{
					"11": {
						"start": 0.05,
						"stop": 0.05
					}
				}
			}
		}
	]
}"""

def ask_overwrite():
    overwrite = str(raw_input('File already exists.  Continue anyway?  Type "continue" or "abort" --> '))
    if overwrite.lower() == 'continue' or overwrite.lower() == 'c':
        pass
    else:
        raise IOError("Change the name of the file in the source code.")

def plot_densities(system, step, date_time_stamp, text):
    plt.figure()
    plt.axes([0.25, 0.1, 0.7, 0.85], axisbg="white", frameon=True)

    plt.plot(system.t, system.M_1, label='Predator Density')
    plt.plot(system.t, system.N_1, label='Prey Density')

    plt.ylim(-1., 1.1*system.K_1)
    plt.xlabel('Time')
    plt.ylabel('Population Density')

    for i, t in enumerate(text):
        plt.text(-.32*system.tf, system.K_1*(1-(.05*i)), t)

    # title = 'Changing Parameter: %s = %.5f' % (var_changing, i)
    # plt.title(title)
    plt.legend(loc=0)

    file_ = "%s/graphs/%s/densities_%03d" % (DIRECTORY, date_time_stamp, step)

    if os.path.isfile(file_) and step == 0:
        ask_overwrite()
    plt.savefig(file_, format = 'png')
    plt.close()
    gc.collect()
    print "GRAPH SAVED: %s" % file_

def plot_traits(system, step, date_time_stamp, text):
    plt.figure()
    plt.axes([0.25, 0.1, 0.7, 0.85], axisbg="white", frameon=True)

    plt.plot(system.t, system.m_1, label='Predator Trait')
    plt.plot(system.t, system.n_1, label='Prey Trait')

    plt.ylim(-.5, 15)
    plt.xlabel('Time')
    plt.ylabel('Trait Value')

    for i, t in enumerate(text):
        plt.text(-.32*system.tf, 14*(1-(.05*i)), t)

    # title = 'Changing Parameter: %s = %.5f' % (var_changing, i)
    # plt.title(title)
    plt.legend(loc=0)

    file_ = "%s/graphs/%s/traits_%03d" % (DIRECTORY, date_time_stamp, step)

    if os.path.isfile(file_) and step == 0:
        ask_overwrite()
    plt.savefig(file_, format = 'png')
    plt.close()
    gc.collect()
    print "GRAPH SAVED: %s" % file_

def irange(start, stop, step):
    r = start
    while r <= stop:
        yield r
        r += step

def give_params_avgattack(alpha, tau, A):
    def avgattack(m, n):
        return ((alpha*tau)/sqrt(A))*exp(-(m - n)**2/(2*A))
    return avgattack

def give_params_avg_pred_fitness(eff, d, avgattack):
    def avg_pred_fitness(m, N, n):
        return eff*avgattack(m,n)*N - d
    return avg_pred_fitness

def give_params_avg_prey_fitness(r, K, avgattack):
    def avg_prey_fitness(M, m, N, n):
        return r*(1 - (N/K)) - avgattack(m,n)*M
    return avg_prey_fitness

def give_params_pred_trait_response(eff, A, avgattack):
    def pred_trait_response(N, m, n):
        return avgattack(m,n)*(eff*N*(n-m))/(A)
    return pred_trait_response

def give_params_prey_trait_response(eff, A, avgattack):
    def pred_trait_response(M, m, n):
        return avgattack(m,n)*(eff*M*(n-m))/(A)
    return pred_trait_response

class System:
    def __init__(self, eff_11, alpha_11, sigma_1, sigmaG_1, beta_1, betaG_1, tau_11, d_1, r_1, K_1, y0, tf):
        self.tf = tf
        self.y0 = y0
        self.t = np.linspace(0, self.tf, 400000)
        
        self.eff_11 = eff_11
        self.alpha_11 = alpha_11
        self.tau_11 = tau_11
        self.sigma_1 = sigma_1
        self.sigmaG_1 = sigmaG_1
        self.beta_1 = beta_1
        self.betaG_1 = betaG_1
        self.d_1 = d_1
        self.r_1 = r_1
        self.K_1 = K_1
        self.A_11 = sigma_1**2 + beta_1**2 + tau_11**2
        
        self.avgattack_11 = give_params_avgattack(self.alpha_11, self.tau_11, self.A_11)

        self.pred_1_fitness = give_params_avg_pred_fitness(self.eff_11, self.d_1, self.avgattack_11)
        self.prey_1_fitness = give_params_avg_prey_fitness(self.r_1, self.K_1, self.avgattack_11)

        self.pred_1_trait_response = give_params_pred_trait_response(self.eff_11, self.A_11, self.avgattack_11)
        self.prey_1_trait_response = give_params_prey_trait_response(self.eff_11, self.A_11, self.avgattack_11)

        self.soln = odeint(self.f, self.y0, self.t)
        
        self.M_1 = self.soln[:, 0]
        self.N_1 = self.soln[:, 1]
        self.m_1 = self.soln[:, 2]
        self.n_1 = self.soln[:, 3]
        
    def f(self, y, t):
        M_1 = y[0]
        N_1 = y[1]
        m_1 = y[2]
        n_1 = y[3]

        f0 = M_1*self.pred_1_fitness(m_1, N_1, n_1)
        f1 = N_1*self.prey_1_fitness(M_1, m_1, N_1, n_1)
        f2 = self.sigmaG_1*self.pred_1_trait_response(N_1, m_1, n_1)
        f3 = self.betaG_1*self.prey_1_trait_response(M_1, m_1, n_1)
        return [f0, f1, f2, f3]

def main():
    #data = json.loads(open("config.json"))
    data = json.loads(JSON)

    for set_ in data["system_parameters"]:
        now = datetime.datetime.now()
        date_time_stamp = now.strftime('%y%m%d_%H%M%S')
        os.system("mkdir %s/graphs/%s" % (DIRECTORY, date_time_stamp))

        # Parameter Step
        steps = int(set_["steps"])
        def get_step_and_start(dictionary):
            return(((dictionary["stop"] - dictionary["start"])/steps), float(dictionary["start"]))

        number_of_graphs = (steps+1)*2
        time_needed = AVG_TIME_PER_GRAPH*number_of_graphs/60.
        print "%d graphs will be generated." % (number_of_graphs)
        print("Approximate Time Needed: %.03f minutes\n\n" % time_needed)

        M0_1_dict     = set_["predator"]["initial_values"]["densities"]["1"]
        N0_1_dict     = set_["prey"]["initial_values"]["densities"]["1"]
        m0_1_dict     = set_["predator"]["initial_values"]["traits"]["1"]
        n0_1_dict     = set_["prey"]["initial_values"]["traits"]["1"]

        eff_11_dict   = set_["interaction_parameters"]["efficiencies"]["11"]
        alpha_11_dict = set_["interaction_parameters"]["max_attack_rates"]["11"]
        tau_11_dict   = set_["interaction_parameters"]["specialization"]["11"]

        sigma_1_dict  = set_["predator"]["trait_variances"]["total"]["1"]
        sigmaG_1_dict = set_["predator"]["trait_variances"]["genetic"]["1"]
        d_1_dict      = set_["predator"]["death_rates"]["1"]

        beta_1_dict   = set_["prey"]["trait_variances"]["total"]["1"]
        betaG_1_dict  = set_["prey"]["trait_variances"]["genetic"]["1"]
        r_1_dict      = set_["prey"]["growth_rates"]["1"]
        K_1_dict      = set_["prey"]["carrying_capacities"]["1"]

        ### Parameters Steps
            ### Densities
        (M0_1_step,     M0_1_start)     = get_step_and_start(M0_1_dict)
        (N0_1_step,     N0_1_start)     = get_step_and_start(N0_1_dict)
            ### Traits
        (m0_1_step,     m0_1_start)     = get_step_and_start(m0_1_dict)
        (n0_1_step,     n0_1_start)     = get_step_and_start(n0_1_dict)

        ### Interaction Parameters
        (eff_11_step,   eff_11_start)   = get_step_and_start(eff_11_dict)
        (alpha_11_step, alpha_11_start) = get_step_and_start(alpha_11_dict)
        (tau_11_step,   tau_11_start)   = get_step_and_start(tau_11_dict)
        
        ### Predator Parameters
        (sigma_1_step,  sigma_1_start)  = get_step_and_start(sigma_1_dict)
        (sigmaG_1_step, sigmaG_1_start) = get_step_and_start(sigmaG_1_dict)
        (d_1_step,      d_1_start)      = get_step_and_start(d_1_dict)

        ### Prey Paraemeters            
        (beta_1_step,   beta_1_start)   = get_step_and_start(beta_1_dict)
        (betaG_1_step,  betaG_1_start)  = get_step_and_start(betaG_1_dict)
        (r_1_step,      r_1_start)      = get_step_and_start(r_1_dict)
        (K_1_step,      K_1_start)      = get_step_and_start(K_1_dict)

        time = 0
        # Final time
        tf = set_["final_time"]

        for step in irange(0, steps, 1):
            ts = timeit.default_timer()

            ### Initial Values
                ### Densities
            M0_1     = M0_1_start     + (step*M0_1_step)
            N0_1     = N0_1_start     + (step*N0_1_step)
                ### Traits
            m0_1     = m0_1_start     + (step*m0_1_step)
            n0_1     = n0_1_start     + (step*n0_1_step)

            # Initial Value Vector
            y0       = [M0_1, N0_1, m0_1, n0_1]

            ### Interaction Parameters
            eff_11   = eff_11_start   + (step*eff_11_step)
            alpha_11 = alpha_11_start + (step*alpha_11_step)
            tau_11   = tau_11_start   + (step*tau_11_step)
            
            ### Predator Parameters
            sigma_1  = sigma_1_start  + (step*sigma_1_step)
            sigmaG_1 = sigmaG_1_start + (step*sigmaG_1_step)
            d_1      = d_1_start      + (step*d_1_step)

            ### Prey Paraemeters            
            beta_1   = beta_1_start   + (step*beta_1_step)
            betaG_1  = betaG_1_start  + (step*betaG_1_step)
            r_1      = r_1_start      + (step*r_1_step)
            K_1      = K_1_start      + (step*K_1_step)
            
            system = System(eff_11, alpha_11, sigma_1, sigmaG_1, beta_1, betaG_1, tau_11, d_1, r_1, K_1, y0, tf)

            ### Get parameters in text format for the graphs           
            text = []
            text.append(r"$e_{11} = %.03f$" % system.eff_11)
            text.append(r"$\alpha_{11} = %.03f$" % system.alpha_11)
            text.append(r"$\tau_{11} = %.03f$" % system.tau_11)
            text.append(r"$\sigma_{1} = %.03f$" % system.sigma_1)
            text.append(r"$\sigma_{G,1} = %.03f$" % system.sigmaG_1)
            text.append(r"$\beta_{1} = %.03f$" % system.beta_1)
            text.append(r"$\beta_{G,1} = %.03f$" % system.betaG_1)
            text.append(r"$d_{1} = %.03f$" % system.d_1)
            text.append(r"$r_{1} = %.03f$" % system.r_1)
            text.append(r"$K_{1} = %.03f$" % system.K_1)

            ### plot results            
            plot_densities(system, step, date_time_stamp, text)
            data_time = timeit.default_timer() - ts
            print "Time taken for this data: %.03f\n" % (data_time)
            time += data_time

            ts = timeit.default_timer()

            plot_traits(system, step, date_time_stamp, text)
            data_time = timeit.default_timer() - ts
            print "Time taken for this data: %.03f\n" % (data_time)
            time += data_time

        print "total time taken: %.03f seconds" % (time)
        print "average time per graph: %.03f seconds" % (time/number_of_graphs)
        print date_time_stamp
        print "\a\a\a"


if __name__ == "__main__":
    main()