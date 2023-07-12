#!/usr/bin/env python3

import argparse
import numpy as np
import subprocess
import yaml

# generate parameters from prior distribution
def get_priors():
    mu = 0.00005 + abs(np.random.normal(0, 0.000025))
    s = abs(np.random.normal(0, 0.05))
    flip = 0.05 + abs(np.random.normal(0, 0.0025))
    return mu, s, flip

def gen_seeds(size=8):
    return list(np.random.choice(range(1000), size=size, replace=False))

# edit config.yml with new parameters
def input_params(configfile, outputdir, warlockdir):
    with open(configfile, 'r') as file:
        data = yaml.safe_load(file)
    priors = get_priors()
    
    data['demon_mu_driver_birth'] = priors[0]
    data['demon_s_driver_birth'] = priors[1]
    data['demon_mu_passenger'] = priors[2]
    data['workflow_repo_path'] = warlockdir
    data['workflow_analysis_outdir'] = outputdir
    data['demon_write_clones_file'] = 1
    data['demon_write_demes_file'] = 1
    data['demon_seed'] = gen_seeds()

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
    parser.add_argument('-c','--configfile', type=str, help='path to warlock config.yml file')
    parser.add_argument('-o','--outputdir', type=str, help='path to output directory to be included in config.yml')
    parser.add_argument('-w','--warlockdir', type=str, help='path to local warlock repo folder')
    #execute the parse_args() method
    args = parser.parse_args()
    
    configfile = args.configfile
    outputdir = args.outputdir
    warlockdir = args.warlockdir

    outputfile = input_params(configfile, outputdir, warlockdir)
    run_sim(outputfile, warlockdir)

if __name__ == "__main__":
    main()