
# Deep Learning for Content-based Image Retrieval in Biomedical Applications - Master's Thesis 

This repository contains the implementation of the Content-based Image Retrieval (CBIR) framework for biomedical applications. The project focuses on utilizing deep learning techniques for image retrieval tasks.

The code files are organized as follows:

- The `database` folder: Contains 16 Python files related to the CBIR framework, including indexing images into Redis, feature extraction models, dataset creation, database and indexing structures, K-means clustering, loss definitions, model architectures, image reconstruction, image retrieval, t-SNE visualization, utility functions, and dataset visualization.
- The `script` folder: Contains automated scripts used during the thesis to run multiple methods at once.
- The `data_visualization` folder: Contains various graphs and files summarizing the data, such as samples from each class, histograms, etc.
- The `cms` folder: Contains graphs generated for the models.
- The `excel_results` folder: Contains the accuracies of the diverse methods tested in this work. 

## Requirements

The packages needed to run this code cna be installed using `requirements.txt`.
- pip install -r requirements.txt

## Code Files

Here is a quick description of each code file within the `database` folder:

- `add_images.py`: Code for indexing images into Redis. Requires a pre-trained feature extractor.
- `autoencoders.py`: Contains the architecture and loss functions related to the autoencoder models.
- `classification_acc.py`: Computes metrics for a classification model (note: it was developed early in the thesis and may require modifications for later use).
- `dataset.py`: Elements related to the creation of the dataset, making it usable by training, indexing, and search methods.
- `db.py`: Methods for the database and indexing structures, including initialization, indexing, search, and index training. This file should be run to train the index.
- `kmeans.py`: Methods related to K-means clustering, such as training, loading the model, and cluster analysis.
- `loss.py`: Definitions of all the losses used for training the feature extraction models.
- `models.py`: Code for training a new feature extractor, including architecture initialization, downloading pretrained weights, and implementing training concepts.
- `rec_images.py`: Generates reconstructed images from an autoencoder model.
- `resnet.py` and `vgg.py`: Architectures of the models for the initial implementation of the Autoencoder.
- `retrieve_images.py`: Displays the 10 most similar images to a query image. Requires a pre-trained feature extractor and indexed data.
- `test_accuracy.py`: Obtains the results of the CBIR framework. Similar requirements as the previous file.
- `tsne.py`: Performs t-SNE on the vectors obtained using a chosen feature extractor. Requires pre-computed vectors and indexed data.
- `utils.py`: Collection of functions used throughout the project.
- `visualization.py`: Functions to compute and display different characteristics of the dataset.

Please refer to the individual code files for more detailed information about their content and usage.

## Training of a feature extractor
To train a new feature extractor, the file `models.py`must be run. It supports the following arguments.
- `--num_features`: The size of the feature vectors created by the model (default: 128)
- `--batch_size`: the number of images composing a batch (default: 32)
- `--model`: the backbone architecture of the feature extractor. Supports ResNet50 (resnet), DenseNet121 (default, densenet), VGG19 (vgg), InceptionV3 (inception), KimiaNet (knet), Swin_v2_b (swin), ViT_b_16 (vision), CvT_21 (cvt), ConvNext_tiny (conv), DeiT (deit), VAE (vae), BYOL (byol), a basic contrastive autoencoder (auto) and a ResNet or VGG based autoencoder (vgg16, vgg11, resnet18, resnet50)
- `--weights`: path where to save the weights of the trained model
- `dr_model': flag to add if the model must be combined with the Deep Ranking architecture
- `--num_epochs`: on how many epochs to train the model (default 5)
- `--training_data`: path to where to find the training data. It must be a folder containing one folder per class.
- `--scheduler`: the scheduler to use during training. Supports None (default), Step and Exponential
- `--gpu_id`: the id in the system onto which to train the model (if parallel not activated)
- `--loss`: the loss to use to train the model. Supports: margin (default), proxy_nca_pp, softmax, softtriple for Deep Metric learning. Supports triplet, BCE, contrastive, cosine, infonce (NT-Xent variant) for contrastive learning. For the Autoencoder architectures, the loss is fixed. 
- `--freeze`: to stop the update of the weights of the pre-trained models
- `--generalise`: to train on only half the classes of the dataset (1, 2) or to train the K-means model (3). Default 0
- `--lr`,`--decay`,`--beta_lr`,`--gamma`,`lr_proxies`:different hyperparameters that can be tuned
- `--parallel`: to split the training on all GPUs available on the system (Flag)
- `--load`: specific to the K-means model. If the K-means model must be load rather than re-trained (complete, partial, None)
- `--augmented`: When using a contrastive loss, indicates if the sampling must be made using an Augmented approach (Flag)
- `--non_contrastive`: when using a contrastive loss, indicates that the pairs must only be positive pairs (Flag)


## Indexing of the images
To index the images in the database, the file `add_images.py`must be run. It supports the following arguments.

- `--num_features`: The size of the feature vectors created by the model used as feature extractor (default: 128)
- `--extractor`: the backbone architecture of the feature extractor. Supports ResNet50 (resnet), DenseNet121 (default, densenet), VGG19 (vgg), InceptionV3 (inception), KimiaNet (knet), Swin_v2_b (swin), ViT_b_16 (vision), CvT_21 (cvt), ConvNext_tiny (conv), DeiT (deit), VAE (vae), BYOL (byol), a basic contrastive autoencoder (auto) and a ResNet or VGG based autoencoder (vgg16, vgg11, resnet18, resnet50)
- `--weights`: path where were saved the weights of the trained feature extractor 
- `dr_model`: flag to add if the model is combined with the Deep Ranking architecture
- `--db_name`: the name of the database onto which the files will be added (default: db)
- `--path`: path to where to find the indexing data. It must be a folder containing one folder per class.
- `--rewrite`: indicates if the database must be reinitialized (Flag)
- `--generalise`: to index only half the classes of the dataset (1, 2) or to use the K-means model (3). Default 0

Note that previously to run the file, the Redis server must be activated through the use of the command: redis-server
## Image Retrieval
To retrieve the most similar images to a query, the file `retrieve_images.py`must be run. It supports the following arguments.

- `--num_features`: The size of the feature vectors created by the model used as feature extractor (default: 128)
- `--extractor`: the backbone architecture of the feature extractor. Supports ResNet50 (resnet), DenseNet121 (default, densenet), VGG19 (vgg), InceptionV3 (inception), KimiaNet (knet), Swin_v2_b (swin), ViT_b_16 (vision), CvT_21 (cvt), ConvNext_tiny (conv), DeiT (deit), VAE (vae), BYOL (byol), a basic contrastive autoencoder (auto) and a ResNet or VGG based autoencoder (vgg16, vgg11, resnet18, resnet50)
- `--weights`: path where were saved the weights of the trained feature extractor 
- `dr_model`: flag to add if the model is combined with the Deep Ranking architecture
- `--db_name`: the name of the database onto which to execute the search (default: db)
- `--path`: path to where to find the query or a dataset of queries. In the second case, one query per class in the folder will be randomly selected. 
- `--generalise`: to use only half the classes of the dataset (1, 2) or to use the K-means model (3) in case of the folder of queries. Default 0
- `--nrt_neigh`: the number of similar images to retrieve (default 10)
- `--results_dir`: path to the directory where to save the retrieved images. 

## Framework evaluation
To compute the accuracies obtained by a framework, the file `test_accuracy.py`must be run. It supports the following arguments. 

- `--num_features`: The size of the feature vectors created by the model used as feature extractor (default: 128)
- `--extractor`: the backbone architecture of the feature extractor. Supports ResNet50 (resnet), DenseNet121 (default, densenet), VGG19 (vgg), InceptionV3 (inception), KimiaNet (knet), Swin_v2_b (swin), ViT_b_16 (vision), CvT_21 (cvt), ConvNext_tiny (conv), DeiT (deit), VAE (vae), BYOL (byol), a basic contrastive autoencoder (auto) and a ResNet or VGG based autoencoder (vgg16, vgg11, resnet18, resnet50)
- `--weights`: path where were saved the weights of the trained feature extractor 
- `dr_model`: flag to add if the model is combined with the Deep Ranking architecture
- `--db_name`: the name of the database onto which to execute the search (default: db)
- `--path`: path to where to find the query dataset. 
- `--generalise`: to use only half the classes of the dataset (1, 2) or to use the K-means model (3) in case of the folder of queries. Default 0
- `--measure`: the name of the protocol to use. Supports: all, weighted, random, stat, separated, remove
- `--project_name`: name of the project onto which to compute the accuracy if only wants the results for one. Default None
- `--class_name`: name of the class onto which to ccompute the accuracy
- `--excel_path`: if the protocol is 'separated', path to the excel file in which the results of each class will be written
- `--name`: name of the excel sheet where the results will be written


## Training of the FAISS index
To train the FAISS index in order to later perform approximate search, the file `db.py`must be run. It supports the following arguments.

- `--num_features`: The size of the feature vectors created by the model used as feature extractor (default: 128)
- `--extractor`: the backbone architecture of the feature extractor. Supports ResNet50 (resnet), DenseNet121 (default, densenet), VGG19 (vgg), InceptionV3 (inception), KimiaNet (knet), Swin_v2_b (swin), ViT_b_16 (vision), CvT_21 (cvt), ConvNext_tiny (conv), DeiT (deit), VAE (vae), BYOL (byol), a basic contrastive autoencoder (auto) and a ResNet or VGG based autoencoder (vgg16, vgg11, resnet18, resnet50)
- `--weights`: path where were saved the weights of the trained feature extractor 
- `dr_model`: flag to add if the model is combined with the Deep Ranking architecture
- `--db_name`: the name of the database onto which to execute the training (default: db)
- `--generalise`: to use only half the classes of the dataset (1, 2) or to use the K-means model (3) in case of the folder of queries. Default 0

Note that the database must have already been filled with the vectors of intterest, prior to the training. 

## Contributions

This implementation is based on several other open-source implementations:
- https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch
- https://github.com/euwern/proxynca_pp
- https://github.com/SathwikTejaswi/deep-ranking/blob/master/Code/data_utils.py
- https://github.com/Horizon2333/imagenet-autoencoder
- https://www.geeksforgeeks.org/contractive-autoencoder-cae/?ref=rp
- https://github.com/pytorch/examples/tree/main/vae
- https://github.com/KevinMusgrave/pytorch-metric-learning
- https://lightning-bolts.readthedocs.io/en/0.3.2/self_supervised_models.html?highlight=AMDIM
- https://github.com/stephdef08/tfe2

