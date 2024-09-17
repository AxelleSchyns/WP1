from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
    )
    args = parser.parse_args()

    plt.figure()
    plt.imshow(Image.open(args.inputfile).convert('RGB'))
    plt.axis('off')
    plt.savefig("query_res.png")