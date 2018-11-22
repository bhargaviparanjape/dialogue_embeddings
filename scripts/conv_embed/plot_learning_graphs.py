from matplotlib import pyplot as plt
import argparse,os,sys,json,csv,numpy as np
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## For all iterations with same epoch number average the loss
    parser.add_argument("--log-file", type=str, default="/home/bhargavi/neulab/dialogue_embeddings/output/log.txt")

    args = parser.parse_args()

    content1 = open(args.log_file).read()
    content = "train: Epoch"

    r = re.compile(r"""train: Epoch""")
    result = r.match(content)

