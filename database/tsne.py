import json
import numpy as np
import faiss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import utils
import redis
import time
import seaborn as sns
if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        '--db_name',
        default='db'
    )

    parser.add_argument(
        '--namefig',
        default='tsne'
    )
    parser.add_argument(
        '--generalise',
        action='store_true'
    )
    args = parser.parse_args()

    index = faiss.read_index(args.db_name + '_labeled')
    r = redis.Redis(host='localhost', port='6379', db=0)
    # Retrieve the vectors from the index
    vectors = index.index.reconstruct_n(0, index.ntotal)

    labels = list(range(index.ntotal)) 
    names = []
    
    if args.generalise:
        names = []
        labs = []
        for l in labels:
            v =r.get(str(l) + 'labeled').decode('utf-8')
            v = json.loads(v)
            names.append(utils.get_class(v[0]['name']))
            labs.append(v[1]['label'])
    else:
        for l in labels:
            n = r.get(str(l) + 'labeled').decode('utf-8')
            names.append(utils.get_class(n))    
    
    classes = list(set(names))
    classes.sort()
    conversion = {x: i for i, x in enumerate(classes)}
    int_names = np.array([conversion[n] for n in names])
    # Perform t-SNE on the vectors
    tsne = TSNE(n_components=2, perplexity = 30, method = 'barnes_hut')
    #tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, metric='euclidean')
    
    t = time.time()
    embeddings = tsne.fit_transform(vectors)
    t_fit = time.time() - t
    print(t_fit)

    sns.scatterplot(embeddings[:,0], embeddings[:,1], hue=names, size=0.5)
    plt.savefig(args.namefig + '_sns.png')
    # Visualize the embeddings
    plt.scatter(embeddings[:,0], embeddings[:,1], c=int_names,cmap='viridis', s=1.5, linewidths=1.5, edgecolors='none')
    plt.colorbar()
    plt.savefig(args.namefig + '.png')

    if args.generalise: 
        labs_k = list(set(labs))
        labs_k.sort()
        conversion_k = {x: i for i, x in enumerate(labs_k)}
        int_labs = np.array([conversion_k[n] for n in labs])

        
        plt.scatter(embeddings[:,0], embeddings[:,1], c=int_labs,cmap='viridis', s=1.5, linewidths=1.5, edgecolors='none')
        plt.colorbar()
        plt.savefig(args.namefig + '_k.png',)

