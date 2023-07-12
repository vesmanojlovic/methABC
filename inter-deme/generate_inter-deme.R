#!/usr/local/bin/Rscript

invisible(library(argparse))
invisible(library(demonmeth))

parser = ArgumentParser(
    prog = "inter-deme_methylation",
    description = "R script for generating inter-deme flip matrices from demon outputs"
)
parser$add_argument("simdir", help="path to the simulations output dir")
parser$add_argument("outputfile", help="name of the output file")

argus <- parser$parse_args()
outfname <- argus$outputfile
simdir <- argus$simdir

raw_files <- load_files(simdir)

ogp <- raw_files$output_genotype_properties
demes <- raw_files$demes
clones <- raw_files$clones
coords <- auto_select(demes = demes)

tree <- min_subtree(tree = output_tree(ogp), demeCoords = coords, clones = clones)

mA <- assign_methylation(tree = tree, nb_sites = 50)

methylation_csv(filename = outfname, coords = coords, demes = demes, clones = clones, methyl_list = mA)