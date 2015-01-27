from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp
from scipy.integrate import odeint
from timeit import default_timer
import gc as garbage
import os
import json
from datetime import datetime

AVG_TIME_PER_GRAPH = 3.664
DIRECTORY = "/Users/samuelfleischer/src/general_eco_evo_models/1x1"

LaTeX_VARIABLE_FORMAT = {
    "M0"    : "M_{0",
    "m0"    : "m_{0",
    "sigma" : "\\sigma_{",
    "sigmaG": "\\sigma_{G",
    "d"     : "d_{",

    "N0"    : "N_{0",
    "n0"    : "n_{0",
    "beta"  : "\\beta_{",
    "betaG" : "\\beta_{G",
    "r"     : "r_{",
    "K"     : "K_{",

    "eff"   : "e_{",
    "tau"   : "\\tau_{",
    "alpha" : "\\alpha_{"
}

def ask_overwrite():
    overwrite = str(raw_input('File already exists.  Continue anyway?  Type "continue" or "abort" --> '))
    if overwrite.lower() == 'continue' or overwrite.lower() == 'c':
        pass
    else:
        raise IOError("Change the name of the file in the source code.")

def plot_densities(system, step, date_time_stamp, text):
    plt.figure()
    plt.axes([0.25, 0.1, 0.7, 0.85], axisbg="white", frameon=True)

    for i, value in enumerate(system.M):
        plt.plot(system.t, system.M[value], label="Predator %d Density" % (i+1))
    for i, value in enumerate(system.N):
        plt.plot(system.t, system.N[value], label="Prey %d Density" % (i+1))

    limit = 1.3*max([system.K[subscript] for subscript in system.K])

    plt.ylim(-1., 1.1*limit)
    plt.xlabel('Time')
    plt.ylabel('Population Density')

    for index, text_line in enumerate(text):
        plt.text(-.32*system.tf, limit*(1-(.05*index)), text_line)

    plt.legend(loc=0)

    file_ = "%s/graphs/%s/densities_%03d" % (DIRECTORY, date_time_stamp, step)

    if os.path.isfile(file_) and step == 0:
        ask_overwrite()
    plt.savefig(file_, format = 'png')
    plt.close()
    garbage.collect()
    print "GRAPH SAVED: %s" % file_

def plot_traits(system, step, date_time_stamp, text):
    plt.figure()
    plt.axes([0.25, 0.1, 0.7, 0.85], axisbg="white", frameon=True)

    for i, value in enumerate(system.m):
        plt.plot(system.t, system.m[value], label="Predator %d Trait" % (i+1))
    for i, value in enumerate(system.n):
        plt.plot(system.t, system.n[value], label="Prey %d Trait" % (i+1))

    plt.ylim(-15, 15)
    plt.xlabel('Time')
    plt.ylabel('Trait Value')

    for index, text_line in enumerate(text):
        plt.text(-.32*system.tf, 14*(1-(.1*index)), text_line)

    plt.legend(loc=0)

    file_ = "%s/graphs/%s/traits_%03d" % (DIRECTORY, date_time_stamp, step)

    if os.path.isfile(file_) and step == 0:
        ask_overwrite()
    plt.savefig(file_, format = 'png')
    plt.close()
    garbage.collect()
    print "GRAPH SAVED: %s" % file_

def irange(start, stop, step):
    r = start
    while r <= stop:
        yield r
        r += step

class System:
    def __init__(self, parameters, tf):
        self.tf = tf
        self.t = np.linspace(0, self.tf, 100000)

        self.M0     = parameters["M0"]
        self.m0     = parameters["m0"]
        self.sigma  = parameters["sigma"]
        self.sigmaG = parameters["sigmaG"]
        self.d      = parameters["d"]

        self.N0     = parameters["N0"]
        self.n0     = parameters["n0"]
        self.beta   = parameters["beta"]
        self.betaG  = parameters["betaG"]
        self.r      = parameters["r"]
        self.K      = parameters["K"]

        self.y0 = []
        for i in xrange(0, len(self.M0)):
            self.y0.append(self.M0[str(i+1)])
        for i in xrange(0, len(self.N0)):
            self.y0.append(self.N0[str(i+1)])
        for i in xrange(0, len(self.m0)):
            self.y0.append(self.m0[str(i+1)])
        for i in xrange(0, len(self.n0)):
            self.y0.append(self.n0[str(i+1)])

        self.eff    = parameters["eff"]
        self.tau    = parameters["tau"]
        self.alpha  = parameters["alpha"]

        self.A                     = {}
        for pred_subscript in self.M0:
            for prey_subscript in self.N0:
                interaction_subscript = pred_subscript + prey_subscript
                self.A[interaction_subscript] = self.sigma[pred_subscript]**2 + self.beta[prey_subscript]**2 + self.tau[interaction_subscript]**2

        self.avgattack             = {}
        for pred_subscript in self.M0:
            for prey_subscript in self.N0:
                interaction_subscript = pred_subscript + prey_subscript
                self.avgattack[interaction_subscript] = self.give_params_avgattack(interaction_subscript)

        self.avg_pred_fitness      = {}
        self.pred_trait_response   = {}
        for pred_subscript in self.M0:
            self.avg_pred_fitness[pred_subscript]    = self.give_params_avg_pred_fitness(pred_subscript)
            self.pred_trait_response[pred_subscript] = self.give_params_pred_trait_response(pred_subscript)

        self.avg_prey_fitness      = {}
        self.prey_trait_response   = {}
        for prey_subscript in self.N0:
            self.avg_prey_fitness[prey_subscript]    = self.give_params_avg_prey_fitness(prey_subscript)
            self.prey_trait_response[prey_subscript] = self.give_params_prey_trait_response(prey_subscript)

        self.soln = odeint(self.f, self.y0, self.t)

        self.M = {}; self.N = {}; self.m = {}; self.n = {}
        for i in xrange(0, len(self.M0)):
            index = i
            self.M[str(i+1)] = self.soln[:,index]
        for i in xrange(0, len(self.N0)):
            index = i + len(self.M0)
            self.N[str(i+1)] = self.soln[:,index]
        for i in xrange(0, len(self.m0)):
            index = i + len(self.M0) + len(self.N0)
            self.m[str(i+1)] = self.soln[:,index]
        for i in xrange(0, len(self.n0)):
            index = i + len(self.M0) + len(self.N0) + len(self.m0)
            self.n[str(i+1)] = self.soln[:,index]

    def f(self, y, t):
        M = [0] * len(self.M0)
        for i in xrange(0, len(self.M0)):
            index = i
            M[i] = y[index]

        N = [0] * len(self.N0)
        for i in xrange(0, len(self.N0)):
            index = i + len(self.M0)
            N[i] = y[index]

        m = [0] * len(self.m0)
        for i in xrange(0, len(self.m0)):
            index = i + len(self.M0) + len(self.N0)
            m[i] = y[index]

        n = [0] * len(self.n0)
        for i in xrange(0, len(self.n0)):
            index = i + len(self.M0) + len(self.N0) + len(self.m0)
            n[i] = y[index]

        f = [0] * (len(self.M0) + len(self.N0) + len(self.m0) + len(self.n0))
        for i in xrange(0, len(self.M0)):
            index = i
            f[index] = M[i]*self.avg_pred_fitness[str(i+1)](m[i], N, n)
        for i in xrange(0, len(self.N0)):
            index = i + len(self.M0)
            f[index] = N[i]*self.avg_prey_fitness[str(i+1)](M, m, N[i], n[i])
        for i in xrange(0, len(self.m0)):
            index = i + len(self.M0) + len(self.N0)
            f[index] = self.sigmaG[str(i+1)]*self.pred_trait_response[str(i+1)](N, m[i], n)
        for i in xrange(0, len(self.n0)):
            index = i + len(self.M0) + len(self.N0) + len(self.m0)
            f[index] = self.betaG[str(i+1)]*self.prey_trait_response[str(i+1)](M, m, n[i])
        
        return f

    def give_params_avgattack(self, interaction_subscript):
        A = self.A[interaction_subscript]
        alpha = self.alpha[interaction_subscript]
        tau = self.tau[interaction_subscript]
        def avgattack(m, n):
            return ((alpha*tau)/sqrt(A))*exp(-(m - n)**2/(2*A))
        return avgattack

    def give_params_avg_pred_fitness(self, pred_subscript):
        d = self.d[pred_subscript]
        def avg_pred_fitness(m, N, n):
            fitness_source = 0
            for prey_subscript in self.N0:
                interaction_subscript = pred_subscript + prey_subscript
                eff = self.eff[interaction_subscript]
                avgattack = self.avgattack[interaction_subscript]
                fitness_source += eff*avgattack(m, n[int(prey_subscript)-1])*N[int(prey_subscript)-1]
            return fitness_source - d
        return avg_pred_fitness

    def give_params_avg_prey_fitness(self, prey_subscript):
        r = self.r[prey_subscript]
        K = self.K[prey_subscript]
        def avg_prey_fitness(M, m, N, n):
            fitness_sink = 0
            for pred_subscript in self.M0:
                interaction_subscript = pred_subscript + prey_subscript
                avgattack = self.avgattack[interaction_subscript]
                fitness_sink += avgattack(m[int(pred_subscript)-1], n)*M[int(pred_subscript)-1]
            return r*(1 - (N/K)) - fitness_sink
        return avg_prey_fitness

    def give_params_pred_trait_response(self, pred_subscript):
        def pred_trait_response(N, m, n):
            response = 0
            for prey_subscript in self.N0:
                interaction_subscript = pred_subscript + prey_subscript
                avgattack = self.avgattack[interaction_subscript]
                eff = self.eff[interaction_subscript]
                A = self.A[interaction_subscript]
                response += avgattack(m,n[int(prey_subscript)-1])*(eff*N[int(prey_subscript)-1]*(n[int(prey_subscript)-1]-m))/(A)
            return response
        return pred_trait_response

    def give_params_prey_trait_response(self, prey_subscript):
        def pred_trait_response(M, m, n):
            response = 0
            for pred_subscript in self.M0:
                interaction_subscript = pred_subscript + prey_subscript
                avgattack = self.avgattack[interaction_subscript]
                eff = self.eff[interaction_subscript]
                A = self.A[interaction_subscript]
                response += avgattack(m[int(pred_subscript)-1],n)*(eff*M[int(pred_subscript)-1]*(n-m[int(pred_subscript)-1]))/(A)
            return response
        return pred_trait_response

def main():
    data = json.loads(open("%s/config.json" % DIRECTORY).read())

    for set_ in data["system_parameters"]:
        CONFIG_DESCRIPTIONS_TO_VARIABLES = {
            "M0"    : set_["predator"]["initial_values"]["densities"],
            "m0"    : set_["predator"]["initial_values"]["traits"],
            "sigma" : set_["predator"]["trait_variances"]["total"],
            "sigmaG": set_["predator"]["trait_variances"]["genetic"],
            "d"     : set_["predator"]["death_rates"],

            "N0"    : set_["prey"]["initial_values"]["densities"],
            "n0"    : set_["prey"]["initial_values"]["traits"],
            "beta"  : set_["prey"]["trait_variances"]["total"],
            "betaG" : set_["prey"]["trait_variances"]["genetic"],
            "r"     : set_["prey"]["growth_rates"],
            "K"     : set_["prey"]["carrying_capacities"],

            "eff"   : set_["interaction_parameters"]["efficiencies"],
            "tau"   : set_["interaction_parameters"]["specialization"],
            "alpha" : set_["interaction_parameters"]["max_attack_rates"]
        }
        now = datetime.now()
        date_time_stamp = now.strftime('%y%m%d_%H%M%S')
        os.system("mkdir -p %s/graphs/%s" % (DIRECTORY, date_time_stamp))

        # Parameter Step
        steps = int(set_["steps"])
        def get_step(dictionary):
            return (dictionary["stop"] - dictionary["start"])/steps

        number_of_graphs = (steps+1)*2
        time_needed = AVG_TIME_PER_GRAPH*number_of_graphs/60.
        print "%d graphs will be generated." % (number_of_graphs)
        print("Approximate Time Needed: %.03f minutes\n\n" % time_needed)


        starts_and_steps = {}

        for variable, dict_ in CONFIG_DESCRIPTIONS_TO_VARIABLES.iteritems():    
            starts_and_steps[variable] = {}
            for subscript, parameter_range in dict_.iteritems():
                starts_and_steps[variable][subscript] = {}
                step = get_step(parameter_range)
                starts_and_steps[variable][subscript]["start"] = parameter_range["start"]
                starts_and_steps[variable][subscript]["step"] = step

        time = 0
        # Final time
        tf = set_["final_time"]

        for step in irange(0, steps, 1):
            ts = default_timer()

            parameters = {}
            for variable, dict_ in starts_and_steps.iteritems():
                parameters[variable] = {}
                for subscript, parameter_range in dict_.iteritems():
                    parameters[variable][subscript] = parameter_range["start"] + (step*parameter_range["step"])
            
            system = System(parameters, tf)

            ### Get parameters in text format for the graphs           
            text = []
            for variable in parameters:
                for subscript, value in parameters[variable].iteritems():
                    text.append(r"$%s%s}= %.03f$" % (LaTeX_VARIABLE_FORMAT[variable], subscript, value))

            ### plot results            
            plot_densities(system, step, date_time_stamp, text)
            data_time = default_timer() - ts
            print "Time taken for this data: %.03f\n" % (data_time)
            time += data_time

            ts = default_timer()

            plot_traits(system, step, date_time_stamp, text)
            data_time = default_timer() - ts
            print "Time taken for this data: %.03f\n" % (data_time)
            time += data_time

        print "total time taken: %.03f seconds" % (time)
        print "average time per graph: %.03f seconds" % (time/number_of_graphs)
        print date_time_stamp
        print "\a\a\a"

if __name__ == "__main__":
    main()