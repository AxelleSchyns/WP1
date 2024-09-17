from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
    )
    args = parser.parse_args()
    extractor_name = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)','(m)','(n)','(o)','(p)','(q)','(r)','(s)','(t)','(u)','(v)','(w)','(x)','(y)','(z)']
    search_strings = ['query', 'resnet', 'deit']
    inputs = []
    folds = os.listdir(args.input_dir)
    folds.sort()
    # Walk through the directory
    for search_string in search_strings:

        for fold in folds:
            for files in os.listdir(os.path.join(args.input_dir, fold)):
                if search_string in files:
                    inputs.append(os.path.join(args.input_dir, fold, files))
    print(len(inputs))
    plt.figure()

    for i in range(1, len(inputs)+1):
            plt.subplot(3,6,i)
            plt.axis('off')
            plt.imshow(Image.open(inputs[i-1]))
            if i < 7:
                plt.title(extractor_name[i-1], fontsize=8)
        
    plt.subplots_adjust(wspace=0, hspace=0)      
    plt.savefig(os.path.join(args.input_dir, "sup_uliege.png"))
