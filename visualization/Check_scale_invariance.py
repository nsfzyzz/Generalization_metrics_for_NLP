import weightwatcher as ww
import matplotlib.pyplot as plt
import numpy as np
import powerlaw

print(ww.__file__)

def generate_average_pdfs(mu = 1, num_ESDs = 10):
    
    evals_all = []
    for esd_index in range(num_ESDs):
        
        N, M = 1000, 500
        Q = N / M
        #W = np.random.normal(0,1,size=(M,N))
        W=np.random.pareto(mu,size=(N,M))
        # X shape is M x M
        X = (1/N)*np.dot(W.T,W)
        evals = np.linalg.eigvals(X)

        # This mimics the usual behavior of NNs, which do not have extremely large evals
        # One can change the constant here to see how the exponential truncation changes with the scale
        evals = evals/10000
        evals = evals[5:]
        
        evals_all.append(evals)
    
    evals_all = np.concatenate(evals_all)
    evals_all = np.sort(evals_all)
    
    return evals_all


num_exps = 10
alphas_final = []
alphas_final_std = []
lambdas_final = []
lambdas_final_std = []
lambdas_adjusted_final = []
lambdas_adjusted_std_final = []
lambdas_adjusted2_final = []
lambdas_adjusted2_std_final = []
mu_list = np.linspace(1.0, 3.0, 20)
#mu_list = [1.0, 3.0]

for mu in mu_list:
    alphas = []
    lambdas = []
    lambdas_adjusted = []
    lambdas_adjusted2 = []
    for exp in range(num_exps):
        evals = generate_average_pdfs(mu = mu)
        print('The number of eigenvalues is')
        print(evals.shape)
        
        watcher = ww.WeightWatcher(model=None)
        alpha, xmin, xmax, D, sigma, num_pl_spikes, best_fit, exponent, sigma_var = \
            watcher.fit_powerlaw(evals, xmax = np.max(evals), plot=False, distribution='truncated_power_law')
        alphas.append(alpha)
        lambdas.append(exponent)
        lambdas_adjusted.append(exponent*xmin)
        lambdas_adjusted2.append(exponent*xmax)
    
    alphas_final.append(np.mean(alphas))
    lambdas_final.append(np.mean(lambdas))
    lambdas_adjusted_final.append(np.mean(lambdas_adjusted))
    lambdas_adjusted2_final.append(np.mean(lambdas_adjusted2))
    
    alphas_final_std.append(np.std(alphas))
    lambdas_final_std.append(np.std(lambdas))
    lambdas_adjusted_std_final.append(np.std(lambdas_adjusted))
    lambdas_adjusted2_std_final.append(np.std(lambdas_adjusted2))
    
alphas_final = np.array(alphas_final)
lambdas_final = np.array(lambdas_final)
lambdas_adjusted_final = np.array(lambdas_adjusted_final)
lambdas_adjusted2_final = np.array(lambdas_adjusted2_final)

alphas_final_std = np.array(alphas_final_std)
lambdas_final_std = np.array(lambdas_final_std)
lambdas_adjusted_std_final = np.array(lambdas_adjusted_std_final)
lambdas_adjusted2_std_final = np.array(lambdas_adjusted2_std_final)

fig, ax = plt.subplots()
ax.plot(mu_list, alphas_final)
ax.set_xlabel("mu: ground-truth powerlaw coefficient", fontsize = 16)
ax.set_ylabel("E-TPL beta", fontsize = 16)
ax.fill_between(mu_list, alphas_final - alphas_final_std, alphas_final + alphas_final_std, alpha=0.3)
plt.savefig('E-TPL-beta.pdf', format='pdf')
plt.clf()

fig, ax = plt.subplots()
ax.plot(mu_list, lambdas_final)
ax.set_xlabel("mu: ground-truth powerlaw coefficient", fontsize = 16)
ax.set_ylabel("E-TPL lambda", fontsize = 16)
ax.fill_between(mu_list, lambdas_final - lambdas_final_std, lambdas_final + lambdas_final_std, alpha=0.3)
plt.savefig('E-TPL-lambda.pdf', format='pdf')
plt.clf()

fig, ax = plt.subplots()
ax.plot(mu_list, lambdas_adjusted_final)
ax.set_xlabel("mu: ground-truth powerlaw coefficient", fontsize = 16)
ax.set_ylabel("E-TPL lambda (adjusted)", fontsize = 16)
ax.fill_between(mu_list, lambdas_adjusted_final - lambdas_adjusted_std_final, lambdas_adjusted_final + lambdas_adjusted_std_final, alpha=0.3)
plt.savefig('E-TPL-lambda-(adjusted).pdf', format='pdf')
plt.clf()

fig, ax = plt.subplots()
ax.plot(mu_list, lambdas_adjusted2_final)
ax.set_xlabel("mu: ground-truth powerlaw coefficient", fontsize = 16)
ax.set_ylabel("E-TPL lambda (adjusted2)", fontsize = 16)
ax.fill_between(mu_list, lambdas_adjusted2_final - lambdas_adjusted2_std_final, lambdas_adjusted2_final + lambdas_adjusted2_std_final, alpha=0.3)
plt.savefig('E-TPL-lambda-(adjusted2).pdf', format='pdf')
plt.clf()
