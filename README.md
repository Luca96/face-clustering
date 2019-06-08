# Face Clustering
Clusterize faces according to facial attributes. FVAB final course project [UL19]. 

> The work presented here, is inspired by the paper: ***"Deep Learning Face Attributes in the Wild "*** by Ziwei Liu et al.

## Project Structure

* **weights/:** contains the weights of the trained model. The model is heavily based on the MobileNetV2 architecture, and is able to infer 37 facial attributes with an **accuracy of 90.95%**.
* **notebook/:** contains the Jupiter Notebook version of the code. Running the code as notebook (either with **Jupiter**, Google **Colab** or **Kaggle** Notebook) is highly suggested. All the work is organized into three sections:
  1. `UL19_Dataset`: Shows how to load and deal with the CelebA dataset. There's a data exploration section which shows how the data is structured and distributed.
  2. `UL19_Training`: Details the model architecture, the training phase (data augmentations, model optimizer, checkpointing), and the testing phase.
  3. `UL19_Clustering`: Puts everything together. You can load your data from multiple sources (CelebA and LFW in the provided code), and let the model infer the attributes. After that, the faces (along with their attributes label) can be grouped into clusters. 
* **py/:** Pure Python version of the notebooks; structured has above.

---

## Usage

1. **Clone** the repository.

2. **(Install libraries):** Only in case you want to test the code (py scripts) on your local machine. The following libraries are required: ***Keras***, ***Tensorflow*** (possibly with GPU compatibly), ***Numpy***, ***Sklearn***, ***Pandas***, ***Matplotlib***, ***OpenCV***, wget (used only to download stuff).

3. **Edit model path:** Before running the code check the model path! In the code I assume that the model is loaded from Google Drive. So you can: 
   * Change every path according to the `weighs` folder of the repo.
   * Or, upload the model on your Google Drive.

---

## Results

Here we show some qualitative results, obtained by dividing in clusters faces taken from the **CelebA** and **LFW** datasets.

![brown_cluster](images/brown_cluster.png)

From a given cluster, we **summarise** it by:

* An **Attribute-occurrence char**: for every attribute, it shows how many times it occurs. This is useful to: (1) spot *noisy-attributes*: the ones with low occurrence frequency, and (2) understand the quality of the cluster by analyzing the distribution of the attributes. 
* An **Eigenface:** it's able to synthesize the prominent attributes (the ones with the greater frequency) at a visual level. A cluster's eigenface carries two type of information at the same time. It shows what are the principal (more common) attributes within that cluster, and implicitly their occurrence frequency (an eigenface vary according to the attributes's frequency).

![blonde_cluster](images/blonde_cluster.png)

In our approach, we first **infer the facial attributes** (in this case only ten are selected), for grouping them through the **K-Means clustering** method. 

Here we show a comparison of the performance of common clustering techniques taken from [sklean library](https://scikit-learn.org/stable/modules/clustering.html). The chart shows the **silhouette score** versus the **number of clusters**.

![clustering_comparison](images/clustering_comparison.png)

 Note that: DBSCAN and OPTICS algorithm infers the number of clusters, automatically.