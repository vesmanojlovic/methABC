#!/usr/local/bin/Rscript

invisible(library(argparse))
invisible(library(demonmeth))

parser = ArgumentParser(
    prog = "intra-deme_methylation",
    description = "R script for generating intra-deme methylation arrays from demon outputs"
)
parser$add_argument("simdir", help="path to the simulations output dir")
parser$add_argument("outputfile", help="name of the output file")

argus <- parser$parse_args()
outfname <- argus$outputfile
simdir <- argus$simdir

raw_files <- load_files(simdir)

tree <- output_tree(output_genotype_properties = raw_files$output_genotype_properties)
demes_data <- raw_files$demes
clones_data <- raw_files$clones

res <- single_deme_avg(tree, demes_data, clones_data, 100, filename=outfname)