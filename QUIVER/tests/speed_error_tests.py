#!/usr/bin/env python3

import torch
import time
import argparse
from scipy.stats import truncnorm
import os
from datetime import datetime

# our python package
import quiver_cpp

##############################################################################
##############################################################################

def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

##############################################################################
##############################################################################

def quantize(vec, sqv):
    
    buckets = torch.bucketize(vec, sqv)
   
    up = torch.take(sqv, buckets)
    down = torch.take(sqv, torch.clip(buckets - 1, min=0))
    
    p = (up - vec) / (up - down)
    r = torch.rand(p.numel(), device=device)
    
    return down + (up - down) * (p < r)
    
##############################################################################
##############################################################################

def calc_vNMSE(vec, sqv):
    buckets = torch.bucketize(vec, sqv)  # buckets is the first that is larger or equal than!

    up = sqv[buckets]  # torch.take(sqv, buckets)
    down = sqv[buckets - 1]  # torch.take(sqv, torch.clip(buckets - 1, min=0))

    return torch.sum((up - vec) * (vec - down)) / torch.norm(vec, 2) ** 2


##############################################################################
##############################################################################

def quiver_exact(svec, s): return quiver_cpp.quiver_exact(svec, s)
def quiver_exact_accelerated(svec, s): return quiver_cpp.quiver_exact_accelerated(svec, s)

##############################################################################
##############################################################################

exact_algs = {}

exact_algs["quiver_exact"] = {}
exact_algs["quiver_exact"]['alg'] = quiver_exact
exact_algs["quiver_exact"]['description'] = "call: quiver_cpp.quiver_exact(svec, s); svec is the sorted vector, s is the number of quantization values"

exact_algs["quiver_exact_accelerated"] = {}
exact_algs["quiver_exact_accelerated"]['alg'] = quiver_exact_accelerated
exact_algs["quiver_exact_accelerated"]['description'] = "call: quiver_cpp.quiver_exact_accelerated(svec, s); svec is the sorted vector, s is the number of quantization values"

##############################################################################
##############################################################################

approx_algs = {}

approx_algs["quiver_approx"] = {}
approx_algs["quiver_approx"]['alg'] = quiver_cpp.quiver_approx
approx_algs["quiver_approx"]['description'] = "call: quiver_cpp.quiver_approx(svec, s, m); svec is the sorted vector, s is the number of quantization values, m is the number of bins"

##############################################################################
##############################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""
    Simulations of Optimal Adaptive Vector Quantization paper.
    Running examples:
    $ python3 speed_error_tests.py --verbose --type exact --folder speed_error --numseeds 5
    $ python3 speed_error_tests.py --verbose --type approx --folder speed_error --numseeds 5

    see more options by running: $ python3 speed_error_tests.py --help
    """,
    formatter_class=argparse.RawTextHelpFormatter)

    ### verbosity
    parser.add_argument('--verbose', default=False, action='store_true', help='detailed progress')

    ### seed
    parser.add_argument('--numseeds', default=3, type=int, help='Number of random seeds per run')

    ### path to result folder
    parser.add_argument('--folder', default='speed_error', type=str, help='results folder')

    ### verbosity
    parser.add_argument('--update_fn', default=None, type=str, help='update results file of specific alg, see args.alg')

    ### simulation type
    parser.add_argument('--type', default='exact', choices=['exact', 'approx'], help='which simulations to run')

    ### specific alg?
    parser.add_argument('--alg', default=None, help='specific alg to run/update, to update results see args.update_fn')

    ### specific distribution
    parser.add_argument('--dist', default=None, choices=['lognormal', 'normal', 'exponential', 'truncnorm', 'weibull'], help='specific dist to run')

    args = parser.parse_args()

    if not os.path.isdir('results'):
        os.mkdir('results')

    if not os.path.isdir('results/{}'.format(args.folder)):
        os.mkdir('results/{}'.format(args.folder))

    if args.update_fn and not args.alg:
        raise RuntimeError('set args.update without specific args.alg')

    seeds = [42, 104, 78, 45, 23, 38, 62, 101, 235, 1001]
    vec_log2_dims = [10, 12, 14, 16, 18, 20, 22, 24]
    nbits_range = range(2, 7)
    m_range = [0] if args.type == 'exact' else [100, 200, 400, 600, 800, 1000]

    algs = exact_algs if args.type == 'exact' else approx_algs

    ### device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    distributions = {
        'exact': ['lognormal', 'normal', 'exponential', 'truncnorm', 'weibull'],
        'approx': ['lognormal', 'normal', 'exponential', 'truncnorm', 'weibull']
    }

    if args.dist:
        distributions_list = [args.dist]
    else:
        distributions_list = distributions[args.type]

    # results dict structure is: results[alg][distribution][vec_dim][nbits][m]
    results = {}
    global_start = datetime.now()

    for seed_idx in range(args.numseeds):
        seed_start = datetime.now()
        seed = seeds[seed_idx]
        torch.cuda.empty_cache()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        ### np.random.seed(seed)
        ### random.seed(seed)

        if args.verbose:
            print('******* seed {} (#{}) *******'.format(seed, seed_idx+1))

        sampled_vecs = {}
        d_max = 2**vec_log2_dims[-1]
        for distribution in distributions_list:
            if distribution == 'lognormal':
                sampled_vecs[distribution] = torch.distributions.LogNormal(0, 1).sample([d_max]).view(-1).to(device)
            elif distribution == 'normal':
                sampled_vecs[distribution] = torch.distributions.Normal(0, 1).sample([d_max]).view(-1).to(device)
            elif distribution == 'exponential':
                sampled_vecs[distribution] = torch.distributions.exponential.Exponential(torch.Tensor([1])).sample([d_max]).view(-1).to(device)
            elif distribution == 'truncnorm':
                sampled_vecs[distribution] = torch.tensor(truncated_normal([d_max]), device=device)
            elif distribution == 'weibull':
                sampled_vecs[distribution] = torch.distributions.weibull.Weibull(torch.tensor([1.0]),torch.tensor([1.0])).sample([d_max]).view(-1).to(device)

        if args.alg:
            if args.alg in algs.keys():
                algs_list = [args.alg]
            else:
                raise RuntimeError("Unknown {} alg in {} algs list ({})".format(args.alg, args.type, algs.keys()))
        else:
            algs_list = list(algs.keys())


        for alg in algs_list:
            if alg not in results.keys():
                results[alg] = {}

            for distribution in distributions_list:

                if distribution not in results[alg].keys():
                    results[alg][distribution] = {}

                for vec_dim in vec_log2_dims:

                    ### params
                    d = 2**vec_dim

                    vec = sampled_vecs[distribution][:2 ** vec_dim].clone()
                    if vec.numel() != 2 ** vec_dim:
                        raise RuntimeError('vec length is not equal {} (2**{})'.format(2 ** vec_dim, vec_dim))

                    if alg == 'quiver_exact_accelerated' and d >= 2**25:
                        print('*** Skipping quiver_exact_accelerated with d={} (d>=2^25) (determined over 64GB RAM) , uncomment the line below to enable it ***'.format(d))
                        continue

                    if d not in results[alg][distribution].keys():
                        results[alg][distribution][d] = {}

                    for nbits in nbits_range:
                        ### params
                        s = 2**nbits

                        if s not in results[alg][distribution][d].keys():
                            results[alg][distribution][d][s] = {}

                        for m in m_range:

                            if m not in results[alg][distribution][d][s].keys():
                                results[alg][distribution][d][s][m] = {}
                                results[alg][distribution][d][s][m]['sort_time[ms]'] = []
                                results[alg][distribution][d][s][m]['sqv_time[ms]'] = []
                                results[alg][distribution][d][s][m]['quantize_time[ms]'] = []
                                results[alg][distribution][d][s][m]['nmse'] = []

                            # ignore m for exact simulation, m is used as a place holder to maintain structure
                            if args.type == 'approx' and (m < s or m > d):
                                continue

                            if torch.cuda.is_available():
                                start_gpu = torch.cuda.Event(enable_timing=True)
                                end_gpu = torch.cuda.Event(enable_timing=True)

                                start_gpu.record()
                            ### sort vector and move to cpu
                            svec, _ = torch.sort(vec)
                            svec = svec.double().to('cpu')

                            if torch.cuda.is_available():
                                end_gpu.record()
                                torch.cuda.synchronize()
                                results[alg][distribution][d][s][m]['sort_time[ms]'].append(start_gpu.elapsed_time(end_gpu))
                            else:
                                results[alg][distribution][d][s][m]['sort_time[ms]'].append(None)

                            try:
                                ### invoke and time
                                start = time.time_ns()
                                if args.type == 'exact':
                                    sqv = algs[alg]['alg'](svec, s)
                                else:
                                    sqv = algs[alg]['alg'](svec, s, m)
                                end = time.time_ns()

                                results[alg][distribution][d][s][m]['sqv_time[ms]'].append((end - start)/1000000)

                                if torch.cuda.is_available():
                                    start_gpu.record()
                                dvec = quantize(vec, sqv.to(device))

                                if torch.cuda.is_available():
                                    end_gpu.record()
                                    torch.cuda.synchronize()
                                    results[alg][distribution][d][s][m]['quantize_time[ms]'].append(start_gpu.elapsed_time(end_gpu))
                                else:
                                    results[alg][distribution][d][s][m]['quantize_time[ms]'].append(None)

                                nmse = calc_vNMSE(vec.to(device), sqv.to(device))
                                results[alg][distribution][d][s][m]['nmse'].append(nmse.item())

                                if args.verbose:
                                    print('{}, {}, d={}, s={}, m={}, nmse={:.5f}, time[ms]={:.2f}'.format(alg, distribution, d, s, m, nmse, (end - start)/1000000))

                            except Exception as error:
                                print(error)
                                results[alg][distribution][d][s][m]['sqv_time[ms]'].append(None)
                                results[alg][distribution][d][s][m]['quantize_time[ms]'].append(None)
                                results[alg][distribution][vec_dim][d][s][m]['nmse'].append(None)

                                if args.verbose:
                                    print('{}, {}, d={}, s={}, m={}'.format(alg, distribution, d, s, m))
                                    print("Failed")

            # temp results snapshot
            torch.save(results,
            './results/' + args.folder + '/' + 'results_{}_{}.pt'.format(args.type, datetime.now().strftime("%m%d%Y%H%M%S")))

        if args.verbose:
            print('****** End of round {}: {} seconds ******'.format(seed_idx+1, datetime.now()-seed_start))

    # save/update results
    if args.update_fn:
        orig_results = torch.load(args.update_fn, map_location='cpu')
        orig_results[args.alg] = {}
        orig_results[args.alg] = results[args.alg]
        results = {}
        results = orig_results

    if args.dist:
        torch.save(results, './results/' + args.folder + '/' + 'results_{}_{}.pt'.format(args.type, args.dist))
    else:
        torch.save(results, './results/' + args.folder + '/' + 'results_{}.pt'.format(args.type))

    print('{} simulations are finished (time[sec]={})'.format(args.type, (datetime.now() - global_start)))
