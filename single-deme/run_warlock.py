#!/usr/bin/env python3

import argparse
import numpy as np
import os
import subprocess
import yaml

from scipy.stats import wasserstein_distance

# generate parameters from prior distribution
def get_priors():
    mu = np.random.uniform(0.00001, 0.001)
    s = np.random.uniform(0, 0.4)
    return mu, s

# edit config.yml with new parameters
def input_params(inputfile, outputdir, warlockdir):
    with open(inputfile, 'r') as file:
        data = yaml.safe_load(file)
    priors = get_priors()
    
    data['demon_mu_driver_birth'] = priors[0]
    data['demon_s_driver_birth'] = priors[1]
    data['workflow_repo_path'] = warlockdir
    data['workflow_analysis_outdir'] = outputdir

    if outputdir[-1] == '/':
        outputfile = outputdir + 'config.yml'
    else:
        outputfile = outputdir + '/config.yml'
    
    with open(outputfile, 'w') as file:
        yaml.dump(data, file)
    return outputfile

# run warlock using given config file
def run_sim(config_file_path, warlockdir):
    command = ['bash', 'warlock.sh', '-c', config_file_path, '-e', 'local']
    subprocess.run(command, cwd=warlockdir)
    return

def main():
    # get input and output file paths
    parser = argparse.ArgumentParser(description='Run simulations for intra-deme selection ABC')
    parser.add_argument('inputfile', type=str, help='path to warlock config.yml file')
    parser.add_argument('outputdir', type=str, help='path to output directory to be included in config.yml')
    parser.add_argument('warlockdir', type=str, help='path to local warlock repo folder')
    #execute the parse_args() method
    args = parser.parse_args()
    
    inputfile = args.inputfile
    outputdir = args.outputdir
    warlockdir = args.warlockdir

    outputfile = input_params(inputfile, outputdir, warlockdir)
    run_sim(outputfile, warlockdir)

if __name__ == "__main__":
    main()