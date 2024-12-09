import matplotlib.pyplot as plt
import numpy as np
import torch
from warnings import warn

def readata(file_path, s, m, metric):
    data = torch.load(file_path)
    speedata = {}
    dims = []
    dists = []
    algs = []

    for algorithm, distributions in data.items():
        for distribution, dimensions in distributions.items():
            if distribution not in speedata:
                speedata[distribution] = {}
            speedata[distribution][algorithm] = []
            for dimension, quantizations in dimensions.items():
                speed_values = quantizations[s][m].get(metric, [])
                if speed_values:
                    dims.append(dimension)
                    dists.append(distribution)
                    speedata[distribution][algorithm].append(np.mean(speed_values))  
    dims = sorted(set(dims))
    dists = sorted(set(dists))
    return speedata, dims, dists, algs
                

# Fixed quantization value
s = 16
m = 400
apx_file_path = "results_approx.pt"
exact_file_path = "results_exact.pt"

try:
    apxspeedata, dims, dists, apxalgs = readata(apx_file_path, s, m, "sqv_time[ms]")
except:
    warn(apx_file_path + ' does not exist. Skipping approx QUIVER')
    apxspeedata = None
try:
    exactspeedata, dims, dists, exactalgs = readata(exact_file_path, s, 0, "sqv_time[ms]")
except:
    warn(exact_file_path + ' does not exist. Skipping exact QUIVER')
    exactspeedata = None
    
if apxspeedata is None and exactspeedata is None:
    raise Exception("No data is found")

for distribution in dists:
    plt.figure(figsize=(8, 6))
    if exactspeedata is not None:            
        for algorithm in exactspeedata[distribution].keys():        
            plt.plot(dims, exactspeedata[distribution][algorithm], label=algorithm)
    if apxspeedata is not None:   
        for algorithm in apxspeedata[distribution].keys():        
            plt.plot(dims, apxspeedata[distribution][algorithm], label=algorithm + ' $m = $' + str(m))
        
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    plt.title(f"Time vs Dimension for {distribution} (Quantization levels ($s$) = {s})")
    plt.xlabel("Dimension ($d$)")
    plt.ylabel("Time [ms]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
try:        
    apxspeedata, dims, dists, apxalgs = readata(apx_file_path, s, m, "nmse")
except:
    pass
try:  
    exactspeedata, dims, dists, exactalgs = readata(exact_file_path, s, 0, "nmse")
except:
    pass    

for distribution in dists:
    plt.figure(figsize=(8, 6))
    if exactspeedata is not None:
        for algorithm in exactspeedata[distribution].keys():        
            plt.plot(dims, exactspeedata[distribution][algorithm], label=algorithm)
    if apxspeedata is not None:   
        for algorithm in apxspeedata[distribution].keys():        
            plt.plot(dims, apxspeedata[distribution][algorithm], label=algorithm + ' $m = $' + str(m))
        
    plt.xscale('log', base=2)    
    plt.title(f"vNMSE vs Dimension for {distribution} (Quantization levels ($s$) = {s})")
    plt.xlabel("Dimension ($d$)")
    plt.ylabel("vNMSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()   