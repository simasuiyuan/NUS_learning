{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "71c65411",
   "metadata": {},
   "source": [
    "# CS5228 Assignment 1b - Clustering &  Association Rule Mining"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "43445bb7",
   "metadata": {},
   "source": [
    "Hello everyone, this assignment notebook covers Clustering (again) and Association Rule Mining (ARM). There are some code-completion tasks and question-answering tasks in this answer sheet. For code completion tasks, please write down your answer (i.e., your lines of code) between sentences that \"Your code starts here\" and \"Your code ends here\". The space between these two lines does not reflect the required or expected lines of code. For answers in plain text, you can refer to [this Markdown guide](https://medium.com/analytics-vidhya/the-ultimate-markdown-guide-for-jupyter-notebook-d5e5abf728fd) to customize the layout (although it shouldn't be needed).\n",
    "\n",
    "When you work on this notebook, you can insert additional code cells (e.g., for testing) or markdown cells (e.g., to keep track of your thoughts). However, before the submission, please remove all those additional cells again. Thanks!\n",
    "\n",
    "**Important:** \n",
    "* Remember to rename and save this Jupyter notebook as **A1b_YourName_YourNUSNETID.ipynb** (e.g., **A1b_BobSmith_e12345678.ipynb**) before submission! Failure to do so will yield a penalty of 1 Point.\n",
    "* Remember to rename and save the script file **A1b_script.py** as **A1b_YourName_YourNUSNETID.py** (e.g., **A1b_BobSmith_e12345678.py**) before submission! Failure to do so will yield a penalty of 1 Point.\n",
    "* Submission deadline is Sep 11, 11.59 pm (together with A1a). Late submissions will be penalized by 10% for each additional day.\n",
    "\n",
    "Please also add your nusnet and student id in the code cell below. This is just to make any identification of your notebook doubly sure."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b6781ed8",
   "metadata": {},
   "outputs": [],
   "source": [
    "student_id = 'E0674520'\n",
    "nusnet_id = 'E0674520'"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "668b56a3",
   "metadata": {},
   "source": [
    "Here is an overview over the tasks to be solved and the points associated with each task. The notebook can appear very long and verbose, but note that a lot of parts provide additional explanations, documentation, or some discussion. The code and markdown cells you are supposed to complete are well, but you can use the overview below to double-check that you covered everything.\n",
    "\n",
    "* **1 Clustering Algorithms (30 Points)**\n",
    "    * 1.1 Implementing K-Means++ (12 Points)\n",
    "        * 1.1 a) Initializing Centroids Based on K-Means++ (5 Points)\n",
    "        * 1.1 b) Assigning Data Points to Clusters (4 Points)\n",
    "        * 1.1 c) Updating the Centroids (3 Points)\n",
    "    * 1.2 Questions about Clustering Algorithms (18 Points)\n",
    "        * 1.2 a) Questions about K-Means (6 Points)\n",
    "        * 1.2 b) Interpreting Dendrograms (6 Points)\n",
    "        * 1.2 c) Comparing the Results of Different Clustering Algorithms (6 Points)\n",
    "* **2 Association Rule Mining (20 Points)**\n",
    "    * 2.1 Implementing Apriori Algorithm (10 Points)\n",
    "        * 2.1 a) Create Candidate Itemsets $L_k$ (6 Points)\n",
    "        * 2.1 b) Generate Frequent Itemsets with Apriori Algorithm (4 Points)\n",
    "    * 2.2 Recommending Movies using ARM (10 Points)\n",
    "        * 2.2 a) Compare the Runs A-D and Discuss your Observations! (3 Points) \n",
    "        * 2.2 b) Compare the Runs A-D and Discuss the Results for Building a Recommendation Engine! (3 Points)\n",
    "        * 2.2 c) Sketch a Movie Recommendation Algorithm Based on ARM (4 Points)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "693531cc",
   "metadata": {},
   "source": [
    "## Setting up the Notebook"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e0483fca",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Some more magic so that the notebook will reload external python modules;\n",
    "# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython\n",
    "# This will automatically reload a .py file every time you make changes and save the file\n",
    "%load_ext autoreload\n",
    "%autoreload 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "5875aa66",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import sys\n",
    "\n",
    "from sklearn.cluster import KMeans\n",
    "\n",
    "from efficient_apriori import apriori\n",
    "\n",
    "from src.utils import plot_kmeans_clustering, powerset, merge_itemsets, unique_items, support, confidence, generate_association_rules, show_top_rules\n",
    "\n",
    "np.set_printoptions(precision=2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b583e6e9",
   "metadata": {},
   "source": [
    "**Important:** This notebook also requires you to complete in a separate `.py` script file. This keeps this notebook cleaner and simplifies testing your implementations for us. As you need to rename the file `A1b_script.py`, you also need to edit the import statement below accordingly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "6dea6158",
   "metadata": {},
   "outputs": [],
   "source": [
    "from A1b_mayuan_e0674520 import MyKMeans\n",
    "#from A1b_BobSmith_e12345678 import MyKMeans # <-- you well need to rename this accordingly"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4f3339fa",
   "metadata": {},
   "source": [
    "-------------------"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b802604c",
   "metadata": {},
   "source": [
    "# 1 Clustering\n",
    "\n",
    "In A1a, we also covered the clustering DBSCAN in the context of noise / outlier detection. In this notebook, we focus on K-Means as well as cover questions about clustering in general."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0c5a9a5b",
   "metadata": {},
   "source": [
    "## 1.1 Implementing K-Means++\n",
    "\n",
    "**Important:** The script file `A1b_script.py` contains the skeleton code for your implementation of K-Means++. All the methods you need to complete are in this file."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ea82119",
   "metadata": {},
   "source": [
    "**Loading a Toy Dataset.** For easy testing and debugging your implementation of K-Means++, we provide you with a simple 2-dimensional dataset containing 100 data points. Just by looking at the plot, one can argue that there are six clusters; although that is not important for the testing and debugging. Later, you can try different values for $k$ and visualize the result using a method we provide as well."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "26c39d07",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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2y6XFBbkkDUFIBFmM6taS+T6l0jbbL15Y4oKjmL2DiLPJT175+VxM6MSZXYqDQBiab320Kw+hQVpGW3J1uJ/tghwnnvuOVVUVKigoEDDhg3TX/7yl5jtN27cqGHDhqmgoEAXXnihFi9enKGewu6urOilXt07x23XbEg9u7vT0ofWCwX/n0lD9OqU4S2L/wDArJYbpqLY16rWyRAnTzfGfd1ktlrIxps3y6eoXnvtNc2YMUPPPfecrr76av3qV7/STTfdpD179qhfv37t2tfW1urmm2/WlClTtGzZMr3zzju67777dP755+v222+34DeAneTnuXTrkC+Z2nohnQvi4k2lAYAZoenzZ9bt19Nv/e92P289giJJj66OX5n9kQmXJD1C3dGp/EyyPMD5xS9+oXvuuUf33nuvJGnhwoX67//+by1atEjz589v137x4sXq16+fFi5cKEm65JJL9N577+mpp54iwIEkqahr/BEcyX4L4gAgkvw8lx4YfXHcZIjqA8dNJVl0ZPQ6m27eLA1wGhsbtW3bNj388MNhx8eOHat333034jnV1dUaO3Zs2LFx48ZpyZIlOnv2rDp3bv/lFgwGFQx+sfFhIBBIQe9hR2tq6vX0W/titrHrgjgAiCXeCEq2pnOni6UBzrFjx9TU1KS+ffuGHe/bt68aGhointPQ0BCx/WeffaZjx47J52s/Dzh//nzNmzcvdR2HLYXqP5hhxwVxABBPrBGUbE3nThdbLDJ2ucK/aAzDaHcsXvtIx0Nmz54tv9/f8jh06FAHeww7MlsDZ8boAbZcEAcAHRFK505FZXYnsDTAKS4uVn5+frvRmqNHj7YbpQnxer0R23fq1Em9e0eOat1ut4qKisIecB6zw67lxd3S3BMAyLxQOne0WjVSbo1eWxrgdOnSRcOGDdPatWvDjq9du1YjR46MeM6IESPatX/zzTd1xRVXRFx/g9zB8CwASD26tf8u9HTrbNt07nSxfIrqwQcf1Isvvqhf//rXev/99zVz5kzV1dVp6tSpks5NL337299uaT916lQdPHhQDz74oN5//339+te/1pIlSzRr1iyrfgXYBMOzAHJZaJPNjz5pv9GwP8Ixp7M8wPnmN7+phQsX6qc//amGDBmit99+W2+88YbKysokSfX19aqrq2tpX1FRoTfeeEMbNmzQkCFD9Oijj+qXv/wl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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "X_kmeans = pd.read_csv('data/a1-kmeans-toy-data.csv', header=None).to_numpy()\n",
    "\n",
    "plt.figure()\n",
    "plt.scatter(X_kmeans[:,0], X_kmeans[:,1])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aedb9eb4",
   "metadata": {},
   "source": [
    "#### 1.1 a) Initializing Centroids Based on K-Means++ (5 Points)\n",
    "\n",
    "As we learned in the lecture, K-Means is rather sensitive to the initialization of the clusters. The most common initialization method is **K-Means++** (see lecture slides). Note that K-Means++ is non-deterministic as it picks the next centroids based on probabilities depending on the distances between the data points and the existing centroids. You will implement the K-Means++ initialization in this task.\n",
    "\n",
    "**Implement method `initialize_centroids()` to calculate the initial centroids based on K-Means++!**\n",
    "\n",
    "**Important:** Avoid using loops in the parts of the code you have to complete. If you use loops but the results are correct, there will be some minor deduction of points. Note that we already imported the method `euclidean_distances()` for you, and you can use anything provided by `numpy` (Hint: Check out the method [`np.random.choice()`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html) as it might  come in very handy).\n",
    "\n",
    "With the code cell below, you can check your implementation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "bdcc7ebf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-0.55 -8.16]\n",
      " [-7.8   5.45]\n",
      " [-4.86 -1.23]]\n"
     ]
    }
   ],
   "source": [
    "# Make the randomness \"predictable\" so the result is always the same\n",
    "# (if you remove this line, the output will change for each run)\n",
    "np.random.seed(0) \n",
    "\n",
    "my_kmeans = MyKMeans(n_clusters=3)\n",
    "my_kmeans.initialize_centroids(X_kmeans)\n",
    "\n",
    "print(my_kmeans.cluster_centers_)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "96d90363",
   "metadata": {},
   "source": [
    "The output of the code cell above should be:\n",
    "    \n",
    "```\n",
    "[[-0.55 -8.16]\n",
    " [-7.8   5.45]\n",
    " [-4.86 -1.23]]\n",
    "\n",
    "```\n",
    "\n",
    "an array with $k=3$ rows -- one for each centroid (i.e., cluster center) -- and each row is an array with 2 coordinates since our dataset is just 2-dimensional. If you change the value for $k$, the shape of the array will change accordingly."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5bc86fdf",
   "metadata": {},
   "source": [
    "#### 1.1 b) Assigning Data Points to Clusters (4 Points)\n",
    "\n",
    "In this step, each data point is assigned to its nearest centroid. Calculating distances and finding the smallest values is very easy with `sklearn` or `numpy`. (Hint: You may want to check [`np.argmin`](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html))\n",
    "\n",
    "**Implement method `assign_clusters()` to assign each data point to its nearest centroid!** Calculating distances and finding the smallest values is very easy with `sklearn` or `numpy`. (Hint: You may want to check [`np.argmin`](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html))\n",
    "\n",
    "The method `assign_clusters()` needs update `self.labels_` which is a 1-dimensional array of length $N$ (number of data points). Each element in `self.labels_` is a value ranging from $0$ to $(k-1)$ indicating to which cluster id a data point belongs to. For example, if `self.labels_ = [1, 0, 2, 2, 1]`, the first and the last data point belong the same cluster. The second data point forms its own cluster; the third and forth data points form the third cluster (here $k=3$)\n",
    "\n",
    "With the code cell below, you can check your implementation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "db189a02",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 0 1 2 0 1 2 0 2 1 1 1 2 1 2 0 0 0 0 0 1 1 2 2 2 1 2 0 2 2 2 1 2 1 1 2 1\n",
      " 0 0 0 2 2 0 1 0 1 0 0 1 1 2 2 0 0 0 0 1 2 0 1 2 2 2 1 2 2 0 1 2 0 1 2 0 0\n",
      " 1 0 1 1 0 2 2 0 1 0 2 0 1 1 1 2 2 0 1 0 2 1 0 1 0 1]\n"
     ]
    }
   ],
   "source": [
    "my_kmeans.assign_clusters(X_kmeans)\n",
    "\n",
    "print(my_kmeans.labels_)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3e6c1b40",
   "metadata": {},
   "source": [
    "The output of the code cell above should be:\n",
    "\n",
    "```\n",
    "[1 0 1 2 0 1 2 0 2 1 1 1 2 1 2 0 0 0 0 0 1 1 2 2 2 1 2 0 2 2 2 1 2 1 1 2 1\n",
    " 0 0 0 2 2 0 1 0 1 0 0 1 1 2 2 0 0 0 0 1 2 0 1 2 2 2 1 2 2 0 1 2 0 1 2 0 0\n",
    " 1 0 1 1 0 2 2 0 1 0 2 0 1 1 1 2 2 0 1 0 2 1 0 1 0 1]\n",
    "```\n",
    "\n",
    "Recall that we have 100 data points and set $k=3$. Hence, the length of the array is 100 and each element is either 0, 1, or 2, representing the cluster id/label. Note that the cluster id has no intrinsic meaning, it's only important that points in the same cluster have the same id/label. For example, the result `[1, 1, 0, 2, 1, 2]` would represent the same clustering as `[0, 0, 2, 1, 0, 1]`"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bb8d8f53",
   "metadata": {},
   "source": [
    "#### 1.1 c) Updating the Centroids (4 Points)\n",
    "\n",
    "After the assignment of the data points to clusters, all centroids need to be moved to the average of their respective clusters. Note that the centroids might not change because the assignment made no changes to the clusters and K-Means is done. We already handle this part in the given skeleton of the implementation.\n",
    "\n",
    "**Implement `update_centroids()` to update the centroids (i.e., cluster center) with respect to the new cluster assignments!** As usual, we recommend making good use of `numpy`, or maybe `sklearn`.\n",
    "\n",
    "With the code cell below, you can check your implementation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "bf32c715",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 4.81 -3.95]\n",
      " [-3.66  5.82]\n",
      " [-2.02  2.25]]\n"
     ]
    }
   ],
   "source": [
    "my_kmeans.update_centroids(X_kmeans)\n",
    "\n",
    "print(my_kmeans.cluster_centers_)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b53ddd72",
   "metadata": {},
   "source": [
    "The output of the code cell above should be:\n",
    "\n",
    "```\n",
    "[[ 4.81 -3.95]\n",
    " [-3.66  5.82]\n",
    " [-2.02  2.25]]\n",
    "```\n",
    "\n",
    "Again, the output reflects our setting of $k=3$ and our 2-dimensional dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "90134a33",
   "metadata": {},
   "source": [
    "**Putting it all together (nothing for you to do here!)** With the initialization of the centroids, the assignment to cluster, and the update of the centroids in place, we have everything in place to perform K-Means++ over a dataset X given a choice for k. The method `fit()` below performs K-Means++ using your implementations of the methods `initialize_centroids()`, `assign_clusters()`, and `update_centroids()`. Note that we can exit the main loop once there is no more change in the cluster assignments; if all fails, the algorithm stops after `max_iter` iterations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "dcaf216b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of iterations: 5\n",
      "\n",
      "Final cluster centers:\n",
      "[[-4.8   3.38]\n",
      " [ 6.62  3.43]\n",
      " [-0.25 -8.62]]\n",
      "\n",
      "Final cluster labels:\n",
      "[0 2 0 0 1 1 1 2 0 0 0 0 0 0 0 1 2 2 2 1 0 0 0 0 1 1 0 1 1 0 0 0 1 1 0 0 1\n",
      " 1 1 1 0 1 2 0 2 0 1 1 0 1 0 0 1 2 2 2 0 0 2 0 0 0 0 1 0 0 1 0 0 2 0 0 2 1\n",
      " 0 2 0 0 2 0 0 2 1 1 0 1 0 0 1 0 1 2 1 1 0 0 1 1 1 0]\n",
      "\n"
     ]
    }
   ],
   "source": [
    "np.random.seed(5) \n",
    "\n",
    "my_kmeans = MyKMeans(n_clusters=3).fit(X_kmeans)\n",
    "\n",
    "print('Number of iterations: {}\\n'.format(my_kmeans.n_iter_))\n",
    "print('Final cluster centers:\\n{}\\n'.format(my_kmeans.cluster_centers_))\n",
    "print('Final cluster labels:\\n{}\\n'.format(my_kmeans.labels_))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "67921948",
   "metadata": {},
   "source": [
    "For a final test, you can also compare your implementation with scitkit-learn's implementation [`sklearn.cluster.KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html). Since we adopted the variable and method names from `KMeans` for `MyKMeans`, the code in the cell below is almost identical to the one above."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "bbd07128",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of iterations: 3\n",
      "\n",
      "Final cluster centers:\n",
      "[[-4.8   3.38]\n",
      " [ 6.62  3.43]\n",
      " [-0.25 -8.62]]\n",
      "\n",
      "Final cluster labels:\n",
      "[0 2 0 0 1 1 1 2 0 0 0 0 0 0 0 1 2 2 2 1 0 0 0 0 1 1 0 1 1 0 0 0 1 1 0 0 1\n",
      " 1 1 1 0 1 2 0 2 0 1 1 0 1 0 0 1 2 2 2 0 0 2 0 0 0 0 1 0 0 1 0 0 2 0 0 2 1\n",
      " 0 2 0 0 2 0 0 2 1 1 0 1 0 0 1 0 1 2 1 1 0 0 1 1 1 0]\n",
      "\n"
     ]
    }
   ],
   "source": [
    "np.random.seed(5) \n",
    "\n",
    "sk_kmeans = KMeans(n_clusters=3).fit(X_kmeans)\n",
    "\n",
    "print('Number of iterations: {}\\n'.format(sk_kmeans.n_iter_))\n",
    "print('Final cluster centers:\\n{}\\n'.format(sk_kmeans.cluster_centers_))\n",
    "print('Final cluster labels:\\n{}\\n'.format(sk_kmeans.labels_))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5fdcc53a",
   "metadata": {},
   "source": [
    "**Important:** As the initialization of the centroids has a random component, the result of `KMeans` and `MyKMeans` might generally not be the same. Even if the final centroids are the same, the order might differ. And again, recall from above that, e.g., the labels `[1, 1, 0, 2, 1, 2]` would represent the same clustering as `[0, 0, 2, 1, 0, 1]`. Only with the chosen values for `np.random.seed()` we see the same results. Still, note that the number of iterations differ."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3b7d8912",
   "metadata": {},
   "source": [
    "**Visualization (nothing for you to do here!)** We provide you with auxiliary method `plot_kmeans_clustering()`; you can checkout `src.utils` to have a look at the code of the method. Feel free to go back and run `MyKMeans` or `KMeans` with different values for $k$ (visually we could already tell that $k=5$ is probably the best choice) and/or different random seeds (or not setting any seed at all). Just note that the given expected outcomes will no longer match for other values for $k$ and other seeds. But you can always visualize the result to assess whether the results look meaningful."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "d88d65f4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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gmD21bZt23LqNREREVHlVWGVJy5cuACDwYBvvvkGderUQXR0NCZNmoT09HQsWrTI6r19+vTBunXrTN/b2dmVdXeJiIiIKr0KC+z69OmDPn36mL6vVasWLl68iJUrVxYY2CkUCvj4+JR1F4mIiIieKJVqjV1KSgrc3d0LvC4iIgJeXl6oV68eJk2ahMTExHLoHREREVHlVmEjdo+7evUqvv76ayxevNjqdX379sWwYcNQo0YNXL9+He+//z6eeeYZnDx5EgqFwuI9Go0GGo3G9L1arS7VvhMRERFVBqU+Yjd37tw8mxse/4qMjDS75/bt2+jTpw+GDRuGiRMnWm1/+PDh6N+/P5o0aYIBAwbgzz//xKVLl7Bjx4587/n000/h4uJi+goICCiV10pERERUmQiiKIql2eC9e/dw7949q9cEBQVBqVQCMAZ13bp1Q5s2bbB+/XpIJEWPNevWrYuJEyfirbfesnje0ohdQEAAUlJS4OzsXOTnEREREZUXtVoNFxeXQsUtpT4V6+npCU9Pz0JdGxcXh27duqFVq1ZYt25dsYK6pKQkxMbGwtfXN99rFApFvtO0RERERLaiwjZP3L59G127dkVAQAAWLVqEu3fvIiEhAQkJCWbXNWjQAGFhYQCAtLQ0hISE4MiRI7hx4wYiIiIwYMAAeHp6YvDgwRXxMoiIiIgqjQrbPLFz505cuXIFV65cgb+/v9m53LPDFy9eREpKCgBAKpXizJkzCA0NxYMHD+Dr64tu3bph8+bNcHJyKtf+ExEREVU2pb7G7klQlLlqIiIioopUlLilUuWxIyIiIqLiY2BHREREZCMY2BERERHZCAZ2RERERDaCgR0RERGRjWBgR0RERGQjGNgRERER2QgGdkREREQ2goEdERERkY1gYEdERERkIxjYEREREdkIBnZERERENoKBHREREZGNYGBHREREZCMY2BERERHZCAZ2RERERDaCgR0RERGRjWBgR0RERGQjGNgRERER2QgGdkREREQ2goEdERERkY1gYEdERERkIxjYEREREdkIBnZERERENoKBHREREZGNYGBHREREZCMY2BERERHZCAZ2RERERDaCgR0RERGRjWBgR0RERGQjGNgRERER2QgGdkREREQ2goEdERERkY1gYEdERERkIxjYEREREdkIBnZERERENoKBHREREZGNYGBHRERE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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#plot_kmeans_clustering(X_kmeans, my_kmeans.labels_, my_kmeans.cluster_centers_) # For your implementation\n",
    "plot_kmeans_clustering(X_kmeans, my_kmeans.labels_, sk_kmeans.cluster_centers_) # For sklearn's implementation"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f024914b",
   "metadata": {},
   "source": [
    "You have now completed the implementation of K-Means (more specifically, K-Means++). Well done! We will check how your implementation will perform on a larger dataset and see how it compares to the solutions of your peers :)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a5795131",
   "metadata": {},
   "source": [
    "### 1.2 Questions about Clustering Algorithms"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "913a7726",
   "metadata": {},
   "source": [
    "#### 1.2 a) Questions about K-Means (6 Points)\n",
    "\n",
    "In the table below are 6 statements that are either True or False. Complete the table to specify whether a statement is True or False, and provide a brief explanation for your answer (Your explanation is more important than a simple True/False answer)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1a538d68",
   "metadata": {},
   "source": [
    "This is a markdown cell. Please fill in your answers for (1)~(6).\n",
    "\n",
    "| No. | Statement                                                                                                   | True or False?       | Brief Explanation |\n",
    "|-----|------------------------------------------------------------------------------------------------------------|--------------| ------- |\n",
    "| (1)  | When using K-Means++, then centroids are always at the position of existing data points | False |  only the initial centroids are existing points, following iterations will take mean of the cluster| \n",
    "| (2)  | K-Means++ ensures that the result will not include any empty clusters. | True|  K-Means initialize with existing data points| \n",
    "| (3)  | K-Means, independent of the initialization method, will always converge to a local minimum | True |   The algorithm always converges but not necessarily to global optimum| \n",
    "| (4)  | K-Means++ will always converge to the global optimum. | False |  whether converge to global optimum may depend on initialization of centroids| \n",
    "| (5)  | K-Means++ initialization is more costly than a random initialization of the centroids but generally converges faster. | True |  K-Means++ initialization required computing distance, but can have better performance in practice|\n",
    "| (6)  | K-Means is insensitive to data normalization/standardization -- that is, for the same $k$ and the same initial centroids, K-Means will yield the same clusters where the data is normalized/standardized or not. | True |  for example, given 2D data, one dimension is excessively larger than other, while calculating the distance, the smaller scaled data will barely taken into account for decision making| "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ff6f317a",
   "metadata": {},
   "source": [
    "#### 1.2 b) Interpreting Dendrograms (6 Points)\n",
    "\n",
    "We saw in the lecture that dendrograms are a meaningful way to visualize the hierarchical relationships between the data points with respect to the clustering using AGNES (or any other hierarchical clustering technique). Properly interpreting is important to get a correct understanding of the underlying data.\n",
    "\n",
    "Below are the plots of 6 different datasets labeled A-F. Each dataset contains 30 data points, each with two dimensions."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3190ab0",
   "metadata": {},
   "source": [
    "<img src=\"data/a1-agnes-data-labeled.png\">"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "518bfb74",
   "metadata": {},
   "source": [
    "Below are 6 dendrograms labeled 1-6. These dendograms show the clustering using **AGNES with Single Linkage** for the 6 datasets above, but in a random order."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c77ced8d",
   "metadata": {},
   "source": [
    "<img src=\"data/a1-agnes-dendrogram-labeled.png\">"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "05779eaf",
   "metadata": {},
   "source": [
    "**Find the correct combinations of datasets and dendrograms** -- that is, find for each dataset the dendrogram that visualizes the clustering using AGNES with Single Linkage! Give a brief explanation for each decision! Complete the table below!\n",
    "\n",
    "**Your Answer:**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "20084365",
   "metadata": {},
   "source": [
    "| Dataset | Dendrogram Nr. | Brief Explanation |\n",
    "| ---  | ---   | ---                  |\n",
    "| **A**    | 6 | there are 2 obvious outliers that distance to a majority data distribution where are interpreted at RHS 2 high level clusters|\n",
    "| **B**    | 1 | majority of the points are closed to each other, and other points are gradually distancing from the single dense cluster |\n",
    "| **C**    | 5 | the center point which at the RHS of dendrogram is the furthermost point to the other points that form a ring outer ring that have balanced distance at LHS of dendrogram |\n",
    "| **D**    | 3 | there are clearly 2 clusters, that can be identified by highest level in the dendrogram |\n",
    "| **E**    | 2 | the data points are evenly distributed in the space so can be interpreted balanced dendrogram |\n",
    "| **F**    | 4 | there are clearly 3 clusters, that can be identified by top 3 level in the dendrogram |"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ca507787",
   "metadata": {},
   "source": [
    "#### 1.2 c) Comparing the Results of Different Clustering Algorithms (6 Points)\n",
    "\n",
    "The figure belows shows the 6 different clusterings A-F, each computed over a dataset of 6 unique data points $x_1 x_2, ..., x_6$. The datasets are independent from each other for the 6 clusterings. Each clustering yields 3 clusters. A `1` in the result table indicates that the corresponding data point is part of the corresponding cluster. For example, in Clustering A, the `1` in the bottom-left cell indicates that data point $x_6$ is part of Cluster $C_1$."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9f0ee007",
   "metadata": {},
   "source": [
    "<img src=\"data/a1-clustering-comparison.png\">"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5a5d0003",
   "metadata": {},
   "source": [
    "**For each clustering, decide which algorithm (K-Means, DBSCAN, AGNES) can have produced the clustering!** Use the table below for the answer. If an algorithm could have produced a clustering, just write *OK* in the respective cell of the table. If an algorithm could not have produced a clustering, enter a brief explanation into the respective table cell.\n",
    "\n",
    "**Note:** Beyond all information stated above, there are additional information about the data, the algorithms, and the clusterings given to you!\n",
    "\n",
    "**Your Answer:**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "19145dce",
   "metadata": {},
   "source": [
    "This is a markdown cell. Please fill complete all table cells.\n",
    "\n",
    "|  | K-Means | DBSCAN       | AGNES |\n",
    "|-----|------------------------------------------------------------------------------------------------------------|--------------| ------- |\n",
    "| **Clustering A**  | There is always at least one item in each cluster | OK | OK |\n",
    "| **Clustering B**  | The clusters are non-hierarchical and they do not overlap | The clusters are non-hierarchical and they do not overlap | OK |\n",
    "| **Clustering C**  | Every member belongs to a cluster which is closer to its cluster than any other clusters | The clusters are non-hierarchical and they do not overlap | Every member belongs to at least one cluster |\n",
    "| **Clustering D**  | Every member belongs to a cluster which is closer to its cluster than any other clusters | OK | Every member belongs to at least one cluster |\n",
    "| **Clustering E**  | OK | OK | OK |\n",
    "| **Clustering F**  | The clusters are non-hierarchical and they do not overlap | The clusters are non-hierarchical and they do not overlap | OK |\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "42ec0879",
   "metadata": {},
   "source": [
    "## 2 Association Rule Mining (ARM)\n",
    "\n",
    "Your task is to implement the Apriori Algorithm for finding Association Rules. In more detail, we focus on the **Apriori Algorithm for finding Frequent Itemsets** -- once we have the Frequent Itemsets, we use a naive approach for the association rule. We will provide a small method for that part later.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "49e6af0f",
   "metadata": {},
   "source": [
    "### 2.1 Implementing Apriori Algorithm"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b5c62cd6",
   "metadata": {},
   "source": [
    "#### Toy Dataset\n",
    "\n",
    "The following dataset with 5 transactions and 6 different items is directly taken from the lecture slides. This should make it easier to test your implementation. The format is a list of tuples, where each tuple represents the set of items of an individual transaction. This format can also be used as input for the `efficient-apriori` package."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a6b311d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "transactions_demo = [\n",
    "    ('bread', 'yogurt'),\n",
    "    ('bread', 'milk', 'cereal', 'eggs'),\n",
    "    ('yogurt', 'milk', 'cereal', 'cheese'),\n",
    "    ('bread', 'yogurt', 'milk', 'cereal'),\n",
    "    ('bread', 'yogurt', 'milk', 'cheese')\n",
    "]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bfe3ad90",
   "metadata": {},
   "source": [
    "#### Auxiliary Methods\n",
    "\n",
    "We want you to focus on the Apriori algorithm. So we provide a set of auxiliary functions. Feel free to look at their implementation in the file `data/utils.py`.\n",
    "\n",
    "The method `unique_items()` returns all the unique items across all transactions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "72077b2a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'bread', 'cereal', 'cheese', 'eggs', 'milk', 'yogurt'}"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "unique_items(transactions_demo)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "629cf522",
   "metadata": {},
   "source": [
    "The method `support()` calculates and returns the support for a given itemset and set of transactions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "d86aeb90",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "support(transactions_demo, ('bread', 'milk'))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0e0fce82",
   "metadata": {},
   "source": [
    "The method `confidence()` calculates and returns the confidence for a given association rules and set of transactions. An association rule is represented by a 2-tuple, where the first element represents itemset X and the second element represents items Y (i.e., $X \\Rightarrow Y$)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "7fa0e07a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.75"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confidence(transactions_demo, (('bread',), ('milk',)))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "49823188",
   "metadata": {},
   "source": [
    "The method `merge_itemsets()` merges two given itemsets into one itemset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "a4cf8e68",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "('bread', 'eggs', 'milk')"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "merge_itemsets(('bread', 'milk'), ('bread', 'eggs'))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8667ae69",
   "metadata": {},
   "source": [
    "For your implementation, you can make use of these auxiliary methods wherever you see fit. And that is, of course, strongly recommended, as it makes the programming task much easier. So, let's get started."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a49dbef4",
   "metadata": {},
   "source": [
    "#### 2.1 a) Create Candidate Itemsets $L_k$ (6 Points)\n",
    "\n",
    "Let's assume we have found $F_{k-1}$, i.e., all Frequent Itemsets for size $k-1$. For example $F_1$ is the set of all Frequent Itemsets of size 1, which is simply the set of unique items across all transactions with sufficient support. The next step is now to find $L_k$, all Candidate Itemsets of size $k$. In the lecture, we introduced two methods for this. For this assignment, we focus on the $\\mathbf{F_{k-1} \\times F_{k-1}}$ method -- that is, we use the Frequent Itemsets from the last step to calculate the Candidate Itemsets for the current step.\n",
    "\n",
    "Recall from the lecture that creating $L_k$ involves two main parts:\n",
    "\n",
    "* **Generating** all possible $k$-itemsets from the Frequent Itemsets $F_{k-1}$; and\n",
    "\n",
    "* **Pruning** all $k$-itemsets that cannot be frequent based on the information we already have ($L_k$ should only contain the itemsets for which we indeed calculate the support for)\n",
    "\n",
    "\n",
    "Recall that we also can (and should) **prune** any Candidate Itemsets than cannot possibly also be Frequent Itemsets  based on the information we already have. In other words, the Candidate Itemsets of size $k$ should only contain the itemsets for which we indeed calculate the support for.\n",
    "\n",
    "**Hint:** In the lecture, to make it more illustrative, we first generate all possible Candidate Itemsets and then prune the ones that cannot possibly be frequent. In practice, to save memory space, it's better to check each Candidate Itemset immediately before even adding it to $L_k$. The skeleton code below reflects this. However, if you indeed want to implement pruning as its own step, you're free to do so.\n",
    "\n",
    "**Implement method `generate_Lk()` to calculate the Candidate Itemsets $L_k$ given the Frequent Itemsets $F_{k-1}$!** Note that we walked in detail through an example of this process in the lecture. Below is a code cell that reflects this example to test your implementation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "6413261c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_Lk(Fk_minus_one):\n",
    "\n",
    "    # The code just looks a bit odd since we cannot get an element from a set using indexing\n",
    "    k = len(next(iter(Fk_minus_one))) + 1\n",
    "    \n",
    "    # Initialize as set as a fail safe to avoid any duplicates\n",
    "    Lk = set()\n",
    "    \n",
    "    for itemset1 in Fk_minus_one:\n",
    "        for itemset2 in Fk_minus_one:\n",
    "            \n",
    "            ######################################################################\n",
    "            ### Your code starts here ############################################\n",
    "            if k < 2:\n",
    "                Lk.add(tuple(set(itemset1 + itemset2)))\n",
    "            elif (itemset1 != itemset2) & (len(itemset1)==k-1) & (len(itemset2)==k-1):\n",
    "                cnt = 0\n",
    "                for sub1 in itemset1:\n",
    "                    if sub1 in itemset2:\n",
    "                        cnt += 1\n",
    "                    if cnt == k-2:\n",
    "                        Lk.add(tuple(set(itemset1 + itemset2)))\n",
    "                        break\n",
    "            ### Your code ends here ##############################################\n",
    "            ######################################################################\n",
    "            \n",
    "            pass # Just there so the empty loop does not throw an error\n",
    "    \n",
    "    ######################################################################\n",
    "    ### Your code starts here ############################################\n",
    "    \n",
    "    # MAY ONLY BE REQUIRED IF YOU TREAT PRUNING AS A SEPARATE STEP!!!\n",
    "    \n",
    "    ### Your code ends here ##############################################\n",
    "    ######################################################################\n",
    "    \n",
    "    return Lk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "45e92ac0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "('yogurt', 'bread', 'cheese')\n",
      "('milk', 'cereal', 'cheese')\n",
      "('yogurt', 'milk', 'cheese')\n",
      "('yogurt', 'cereal', 'cheese')\n",
      "('yogurt', 'milk', 'cereal')\n",
      "('milk', 'bread', 'cheese')\n",
      "('yogurt', 'cereal', 'bread')\n",
      "('milk', 'cereal', 'bread')\n",
      "('yogurt', 'milk', 'bread')\n"
     ]
    }
   ],
   "source": [
    "k_itemsets = generate_Lk({\n",
    "    ('bread', 'cereal'), ('bread', 'milk'), ('bread', 'yogurt'), ('cereal', 'milk'),\n",
    "    ('cereal', 'yogurt'), ('cheese', 'milk'), ('cheese', 'yogurt'), ('milk', 'yogurt')\n",
    "})\n",
    "\n",
    "for itemset in k_itemsets:\n",
    "    print(itemset)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8c7ce928",
   "metadata": {},
   "source": [
    "#### 2.1 b) Generate Frequent Itemsets with Apriori Algorithm (4 Points)\n",
    "\n",
    "The method `generate_Lk()` covered the \"Generate\" and \"Prune\" steps of the Apriori Algorithm for finding Frequent Itemsets. Now only the \"Calculate\" and \"Filter\" step is missing. However, with `generate_kplus1_itemsets()` in place and together with the auxiliary methods we provide (see above), putting the Apriori Algorithm together should be pretty straightforward.\n",
    "\n",
    "**Implement `frequent_itemsets_apriori()` to find all Frequent Itemset given a set of transactions and a minimum support of `min_support`!** Again, below is a code cell that reflects this example to test your implementation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "63b0a0dd",
   "metadata": {},
   "outputs": [],
   "source": [
    "def frequent_itemsets_apriori(transactions, min_support):\n",
    "    \n",
    "    # The frequent 1-itemsets are all unique items across all transactions with sufficient support\n",
    "    # The one-liner below simply loops over all uniques items and checks the condition w.r.t. the support\n",
    "    F1 = set([(s,) for s in unique_items(transactions) if support(transactions, (s,)) >= min_support ])\n",
    "    \n",
    "    # If there is not even a single 1-itemset that is frequent, we can just stop here\n",
    "    if len(F1) == 0:\n",
    "        return {}\n",
    "    \n",
    "    # Initialize dictionary with all current frequent itemsets for each size k\n",
    "    # Example: { 1: {(a), (b), (c)}, 2: {(a, c), ...} }\n",
    "    F = { 1: F1 }\n",
    "    \n",
    "    # Find now all frequent itemsets of size 2, 3, 4, ... (sys.maxsize basically mean infinity here)\n",
    "    for k in range(2, sys.maxsize):\n",
    "\n",
    "        Fk = set()\n",
    "        \n",
    "        ########################################################################################\n",
    "        ### Your code starts here ##############################################################\n",
    "\n",
    "        LK = generate_Lk(F[k-1])\n",
    "        if not LK:\n",
    "            break\n",
    "\n",
    "        for candidate in LK:\n",
    "            if support(transactions, candidate) >= min_support:\n",
    "                Fk.add(candidate)\n",
    "                \n",
    "        if not Fk:\n",
    "            break\n",
    "        \n",
    "        ### Your code ends here ################################################################\n",
    "        ########################################################################################\n",
    "                \n",
    "        F[k] = Fk\n",
    "    # Merge the dictionary of itemsets to a single set and return it\n",
    "    # Example: {1: {(a), (b), (c)}, 2: (a, c)} => {(a), (b), (c), (a, c)}\n",
    "    return set.union(*[ itemsets for k, itemsets in F.items() ])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "5ce7be60",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "('bread',)\n",
      "('milk', 'bread')\n",
      "('yogurt',)\n",
      "('yogurt', 'bread')\n",
      "('milk',)\n",
      "('cereal',)\n",
      "('yogurt', 'milk')\n",
      "('milk', 'cereal')\n"
     ]
    }
   ],
   "source": [
    "frequent_itemsets = frequent_itemsets_apriori(transactions_demo, 0.6)\n",
    "for itemset in frequent_itemsets:\n",
    "    print(itemset)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1c5d0094",
   "metadata": {},
   "source": [
    "#### From Frequent Itemsets to Association Rules (nothing for you to do here!)\n",
    "\n",
    "Your implementation so far gives you the Frequent Itemsets in a list of transactions using the Apriori method. This step is typically the most time-consuming one in Association Rule Mining. However, we still have to do the second step and find all Association Rules given the Frequent Itemsets. We saw in the lecture that this can also be done in an efficient manner using the Apriori method to avoid checking all rules.\n",
    "\n",
    "Since this step is typically less computationally expensive, we simply do it the naive way -- that is, we go over all Frequent Itemsets, and check for each Frequent Itemset and which of the Association Rules that can be generated from it has a sufficiently high confidence. With all the auxiliary methods we provide, this becomes trivial to implement, so we simply give you the method `find_association_rules()` below. Note how it uses your implementation of `frequent_itemsets_apriori()`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "6abc8144",
   "metadata": {},
   "outputs": [],
   "source": [
    "def find_association_rules(transactions, min_support, min_confidence):\n",
    "    # Initialize empty list of association rules\n",
    "    association_rules = []\n",
    "    \n",
    "    # Find and loop over all frequent itemsets\n",
    "    for itemset in frequent_itemsets_apriori(transactions, min_support):\n",
    "        if len(itemset) == 1:\n",
    "            continue\n",
    "\n",
    "        # Find and loop over all association rules that can be generated from the itemset\n",
    "        for r in generate_association_rules(itemset):\n",
    "            # Check if the association rule fulfils the confidence requriement\n",
    "            if confidence(transactions, r) >= min_confidence:\n",
    "                association_rules.append(r)\n",
    "                \n",
    "    # Return final list of association rules\n",
    "    return association_rules"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "e2ba7b23",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(('cheese',), ('milk', 'yogurt'))\n",
      "(('cheese', 'milk'), ('yogurt',))\n",
      "(('cheese', 'yogurt'), ('milk',))\n",
      "(('cereal', 'yogurt'), ('milk',))\n",
      "(('cheese',), ('yogurt',))\n",
      "(('cheese',), ('milk',))\n",
      "(('cereal',), ('milk',))\n",
      "(('bread', 'cereal'), ('milk',))\n"
     ]
    }
   ],
   "source": [
    "for rule in find_association_rules(transactions_demo, 0.4, 1.0):\n",
    "    print(rule)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "88167699",
   "metadata": {},
   "source": [
    "#### Comparison with `efficient-apriori` package  (nothing for you to do here!)\n",
    "\n",
    "You can run the apriori algorithm over the demo data to check if your implementation is correct. Try different values for the parameters `min_support` and `min_confidence` and compare the results. Note that the order of the returned association rules might differ between your implementation and the apriori one."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "62eee91f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Rule [('cereal',) => ('milk',)] (support: 0.6, confidence: 1.0, lift: 1.25)\n",
      "Rule [('cheese',) => ('milk',)] (support: 0.4, confidence: 1.0, lift: 1.25)\n",
      "Rule [('cheese',) => ('yogurt',)] (support: 0.4, confidence: 1.0, lift: 1.25)\n",
      "Rule [('bread', 'cereal') => ('milk',)] (support: 0.4, confidence: 1.0, lift: 1.25)\n",
      "Rule [('cereal', 'yogurt') => ('milk',)] (support: 0.4, confidence: 1.0, lift: 1.25)\n",
      "Rule [('cheese', 'yogurt') => ('milk',)] (support: 0.4, confidence: 1.0, lift: 1.25)\n",
      "Rule [('cheese', 'milk') => ('yogurt',)] (support: 0.4, confidence: 1.0, lift: 1.25)\n",
      "Rule [('cheese',) => ('milk', 'yogurt')] (support: 0.4, confidence: 1.0, lift: 1.6666666666666667)\n"
     ]
    }
   ],
   "source": [
    "_, rules = apriori(transactions_demo, min_support=0.4, min_confidence=1.0)\n",
    "\n",
    "for r in rules:\n",
    "    print('Rule [{} => {}] (support: {}, confidence: {}, lift: {})'.format(r.lhs, r.rhs, r.support, r.confidence, r.lift))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "97a70dbc",
   "metadata": {},
   "source": [
    "The `efficient-apriori` provides, of course, a much more efficient and convenient (e.g., keeping track of all the metrics for each rule). And this is why we use this package for finding Association Rules in a real-world dataset below. Still, in its core, `efficient-apriori` implements the same underlying Apriori method to Find Frequent Itemsets (but also to find the Association Rules). If you're interested, further below, you can compare the runtimes of `efficient-apriori` and your implementation. Just don't be too disappointed :)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6443ffda",
   "metadata": {},
   "source": [
    "### 2.2 Recommending Movies using ARM\n",
    "\n",
    "In this task, we look into using Association Rule Mining for recommending movies -- more specifically, recommending movies on physical mediums (Blu-ray, DVD, etc.), assuming that is still a thing nowadays :).\n",
    "\n",
    "**Dataset.** E-commerce sites do not really make their data publicly available, so we do not have any hard real-world dataset. For the context of this assignment, this is of course no problem. What we use here is a popular movie ratings dataset from [MovieLens](https://grouplens.org/datasets/movielens/). This dataset contains user ratings for movies (1-5 stars, incl. half stars, e.g., 3.5). More specifically, we use the [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/) containing 1 Million ratings from ~6,000 users on ~4,000 movies and was released February 2003 -- so do not expect any recent Marvel movies :).\n",
    "\n",
    "While using these ratings allow for more sophisticated recommendation algorithms -- and we will look into some of those in a later lecture -- here we are focusing on Association Rules. This includes that we need to convert this rating dataset into a transaction dataset, where a transaction represents all the movies a user has purchased. We already did this for you making the following assumption: A User has purchased all the movies s/he gave the highest rating. For example, if User A gave a highest rating of 4.5 to any movie, A has purchased all movies A rated with 4.5. This is certainly a simplifying assumption, but perfectly fine for this task here.\n",
    "\n",
    "Let's have a quick look at the data. First, we load the ids and names of all movies into a dictionary. We need this dictionary since our transactions (i.e., the list of movies a user has bought) contains the ids and not the names of the movies. So to actually see the names of movies in the association rules, we need this way to map from a movie's id to its name."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "d934b1eb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1 -> Toy Story\n",
      "2 -> Jumanji\n",
      "3 -> Grumpier Old Men\n",
      "4 -> Waiting to Exhale\n",
      "5 -> Father of the Bride Part II\n"
     ]
    }
   ],
   "source": [
    "# Read file with movies (and der ids) into a pandas dataframe\n",
    "df_movies = pd.read_csv('data/a1-arm-movies.csv', header=None)\n",
    "# Convert dataframe to dictionary for quick lookups\n",
    "movie_map = dict(zip(df_movies[0], df_movies[1]))\n",
    "# Show the first 5 entries as example\n",
    "for movie_id, movie_name in movie_map.items():\n",
    "    print('{} -> {}'.format(movie_id, movie_name))\n",
    "    if movie_id >= 5:\n",
    "        break"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4e553756",
   "metadata": {},
   "source": [
    "No we can load the transactions. Again, a transaction is a user's shopping history, i.e., all the movies the user has bought. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "c3717ccd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shopping history for user 0 (used for Aprior algorithm)\n",
      "(1, 48, 150, 527, 595, 1022, 1028, 1029, 1035, 1193, 1270, 1287, 1836, 1961, 2028, 2355, 2804, 3105)\n",
      "\n",
      "Detailed shopping history for user 0\n",
      "1: Toy Story\n",
      "48: Pocahontas\n",
      "150: Apollo 13\n",
      "527: Schindler's List\n",
      "595: Beauty and the Beast\n",
      "1022: Cinderella\n",
      "1028: Mary Poppins\n",
      "1029: Dumbo\n",
      "1035: Sound of Music, The\n",
      "1193: One Flew Over the Cuckoo's Nest\n",
      "1270: Back to the Future\n",
      "1287: Ben\n",
      "1836: Last Days of Disco, The\n",
      "1961: Rain Man\n",
      "2028: Saving Private Ryan\n",
      "2355: Bug's Life, A\n",
      "2804: Christmas Story, A\n",
      "3105: Awakenings\n"
     ]
    }
   ],
   "source": [
    "shopping_histories = []\n",
    "\n",
    "# Read shopping histories; each line is a comma-separated list of the movies (i.e., their ids!) a user bought\n",
    "with open('data/a1-arm-movie-shopping-histories.csv') as file:\n",
    "    for line in file:\n",
    "        shopping_histories.append(tuple([ int(i) for i in line.strip().split(',') ]))\n",
    "\n",
    "# Show the shopping history of the first user for an example; we need movie_map to get the name of each movie\n",
    "user = 0\n",
    "\n",
    "print('Shopping history for user {} (used for Aprior algorithm)'.format(user))\n",
    "print(shopping_histories[user])\n",
    "print()\n",
    "print('Detailed shopping history for user {}'.format(user))\n",
    "for movie_id in shopping_histories[user]:\n",
    "    print('{}: {}'.format(movie_id, movie_map[movie_id]))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "79366a78",
   "metadata": {},
   "source": [
    "With the dataset loaded, we are ready to find interesting Association Rules. For performance reasons, we use the `efficient_apriori` package -- however, further below there is an optional code cell where you can use your own implementation of the Apriori algorithm, in case you are interested.\n",
    "\n",
    "For added convenience, we provide method `show_top_rules()` which computes the Association Rules using the `efficient-apriori` package, but (a) sorts the rules w.r.t. the specified metric (default: lift), and (b) shows only the top-k rules (default: 5). The method also ensures a consistent output of each Association Rule. Each rule contains the LHS, RHS, as well as the support (s), confidence (c), and lift (l). Feel free to check out the code of method `show_top_rules()` in `src.utils` if anything might be unclear regarding its use.\n",
    "\n",
    "**Run the following 4 code cells and interpret the results below!** All 4 code cells find Association Rules using the `efficient-apriori` package encapsulated in the auxiliary method `show_top_rules()` for convenience. Appreciate how Runs A-B differ with respect to the input parameter of the method calls! Also, note that we call `show_top_rules()` with `id_map=None` at first, so the results will only display the movie ids. Later, you will be asked to run the cells again with `id_map=movie_map` to see the actual names of the movies."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "f2d64405",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=== Total Number of Rules: 68 ===\n",
      "(1221) => (858)  [s: 0.13, c: 0.86, l: 3.51]\n",
      "(858) => (1221)  [s: 0.13, c: 0.55, l: 3.51]\n",
      "(260; 1210) => (1196)  [s: 0.11, c: 0.84, l: 3.43]\n",
      "(1196) => (260; 1210)  [s: 0.11, c: 0.44, l: 3.43]\n",
      "(260; 1196) => (1210)  [s: 0.11, c: 0.58, l: 3.42]\n",
      "(1210) => (260; 1196)  [s: 0.11, c: 0.63, l: 3.42]\n",
      "(1210) => (1196)  [s: 0.12, c: 0.73, l: 2.96]\n",
      "(1196) => (1210)  [s: 0.12, c: 0.51, l: 2.96]\n",
      "(260; 1198) => (1196)  [s: 0.11, c: 0.71, l: 2.88]\n",
      "(1196) => (260; 1198)  [s: 0.11, c: 0.46, l: 2.88]\n",
      "(1196; 1210) => (260)  [s: 0.11, c: 0.87, l: 2.86]\n",
      "\n",
      "Wall time: 145 ms\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "# Run A\n",
    "show_top_rules(shopping_histories, min_support=0.1, min_confidence=0.2, k=10, id_map=None)\n",
    "#show_top_rules(shopping_histories, min_support=0.1, min_confidence=0.2, k=10, id_map=movie_map)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "8b75ea14",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=== Total Number of Rules: 2962203 ===\n",
      "(1132) => (1131)  [s: 0.01, c: 0.64, l: 35.18]\n",
      "(1131) => (1132)  [s: 0.01, c: 0.60, l: 35.18]\n",
      "(1148; 1221) => (745; 858)  [s: 0.01, c: 0.60, l: 28.40]\n",
      "(745; 858) => (1148; 1221)  [s: 0.01, c: 0.48, l: 28.40]\n",
      "(260; 1148; 1196) => (745; 1210)  [s: 0.01, c: 0.39, l: 27.32]\n",
      "(745; 1210) => (260; 1148; 1196)  [s: 0.01, c: 0.70, l: 27.32]\n",
      "(260; 745; 1196) => (1148; 1210)  [s: 0.01, c: 0.49, l: 27.29]\n",
      "(1148; 1210) => (260; 745; 1196)  [s: 0.01, c: 0.56, l: 27.29]\n",
      "(858; 1148) => (745; 1221)  [s: 0.01, c: 0.37, l: 26.54]\n",
      "(745; 1221) => (858; 1148)  [s: 0.01, c: 0.73, l: 26.54]\n",
      "(260; 745; 1148) => (1196; 1223)  [s: 0.01, c: 0.39, l: 25.65]\n",
      "\n",
      "Wall time: 2min 23s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "# Run B\n",
    "show_top_rules(shopping_histories, min_support=0.01, min_confidence=0.2, k=10, id_map=None)\n",
    "#show_top_rules(shopping_histories, min_support=0.01, min_confidence=0.2, k=10, id_map=movie_map)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "563ec2f6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=== Total Number of Rules: 4 ===\n",
      "(1221) => (858)  [s: 0.13, c: 0.86, l: 3.51]\n",
      "(260; 1210) => (1196)  [s: 0.11, c: 0.84, l: 3.43]\n",
      "(1196; 1210) => (260)  [s: 0.11, c: 0.87, l: 2.86]\n",
      "(1196; 1198) => (260)  [s: 0.11, c: 0.84, l: 2.76]\n",
      "\n",
      "Wall time: 138 ms\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "# Run C\n",
    "show_top_rules(shopping_histories, min_support=0.1, min_confidence=0.8, k=10, reverse=True, id_map=None)\n",
    "#show_top_rules(shopping_histories, min_support=0.1, min_confidence=0.8, k=10, reverse=True, id_map=movie_map)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "ee231d29",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=== Total Number of Rules: 122486 ===\n",
      "(745; 1196; 1223) => (260; 1148)  [s: 0.01, c: 0.84, l: 21.17]\n",
      "(745; 1196; 1198) => (260; 1148)  [s: 0.01, c: 0.83, l: 20.89]\n",
      "(745; 1196; 1210) => (260; 1148)  [s: 0.01, c: 0.81, l: 20.55]\n",
      "(720; 1223) => (745; 1148)  [s: 0.01, c: 0.82, l: 15.70]\n",
      "(2571; 2951) => (1201)  [s: 0.01, c: 0.83, l: 15.44]\n",
      "(1; 1223) => (745; 1148)  [s: 0.01, c: 0.80, l: 15.39]\n",
      "(1089; 1196; 1221; 2858) => (260; 296; 858)  [s: 0.01, c: 0.80, l: 15.39]\n",
      "(1089; 1196; 1198; 1221) => (260; 296; 858)  [s: 0.01, c: 0.80, l: 15.34]\n",
      "(260; 1196; 2951) => (1201)  [s: 0.01, c: 0.82, l: 15.27]\n",
      "(858; 2951) => (1201)  [s: 0.01, c: 0.81, l: 15.07]\n",
      "(1196; 2951) => (1201)  [s: 0.01, c: 0.81, l: 14.96]\n",
      "\n",
      "Wall time: 1min 34s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "# Run D\n",
    "show_top_rules(shopping_histories, min_support=0.01, min_confidence=0.8, k=10, reverse=True, id_map=None)\n",
    "#show_top_rules(shopping_histories, min_support=0.01, min_confidence=0.8, k=10, reverse=True, id_map=movie_map)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b376c706",
   "metadata": {},
   "source": [
    "**Optional:** Feel free to uncomment and run the code cell below. It uses your implementation of the Apriori algorithm using the same parameters as Run C. You can use this code to double-check your implementation, but please be aware that it will run longer than the `efficient_apriori` package; although not too long for these parameters. Note that the result will not be in the same format and not sorted, but you can easily eyeball that the results will match the one of Run C above...or at least should :)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "a0565a21",
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<timed exec>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n",
      "\u001b[1;32m<ipython-input-31-d12bd980ae65>\u001b[0m in \u001b[0;36mfind_association_rules\u001b[1;34m(transactions, min_support, min_confidence)\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m     \u001b[1;31m# Find and loop over all frequent itemsets\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 6\u001b[1;33m     \u001b[1;32mfor\u001b[0m \u001b[0mitemset\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mfrequent_itemsets_apriori\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtransactions\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmin_support\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      7\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitemset\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      8\u001b[0m             \u001b[1;32mcontinue\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m<ipython-input-29-bc96c682c3ac>\u001b[0m in \u001b[0;36mfrequent_itemsets_apriori\u001b[1;34m(transactions, min_support)\u001b[0m\n\u001b[0;32m     26\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     27\u001b[0m         \u001b[1;32mfor\u001b[0m \u001b[0mcandidate\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mLK\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 28\u001b[1;33m             \u001b[1;32mif\u001b[0m \u001b[0msupport\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtransactions\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcandidate\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m>=\u001b[0m \u001b[0mmin_support\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     29\u001b[0m                 \u001b[0mFk\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0madd\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcandidate\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     30\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m\\\\fsfshadoopstage\\hadoop_stage\\PEE_joint\\NUS_modules\\CS5344\\Assignments\\cs5228-assignment-1b\\src\\utils.py\u001b[0m in \u001b[0;36msupport\u001b[1;34m(transactions, itemset)\u001b[0m\n\u001b[0;32m     82\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     83\u001b[0m     \u001b[1;31m# Return support count\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 84\u001b[1;33m     \u001b[1;32mreturn\u001b[0m \u001b[0msupport_count\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtransactions\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mitemset\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m/\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtransactions\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     85\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     86\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m\\\\fsfshadoopstage\\hadoop_stage\\PEE_joint\\NUS_modules\\CS5344\\Assignments\\cs5228-assignment-1b\\src\\utils.py\u001b[0m in \u001b[0;36msupport_count\u001b[1;34m(transactions, itemset)\u001b[0m\n\u001b[0;32m     69\u001b[0m     \u001b[1;31m# If so, increment support count\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     70\u001b[0m     \u001b[1;32mfor\u001b[0m \u001b[0mt\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mtransactions\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 71\u001b[1;33m         \u001b[1;32mif\u001b[0m \u001b[0mset\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitemset\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0missubset\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mset\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mt\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     72\u001b[0m             \u001b[0msupport_count\u001b[0m \u001b[1;33m+=\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     73\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "%%time\n",
    "\n",
    "rules = find_association_rules(shopping_histories, 0.1, 0.8)\n",
    "\n",
    "for lhs, rhs in rules:\n",
    "   print('{} => {}'.format(lhs, rhs))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "03dc0961",
   "metadata": {},
   "source": [
    "#### 2.2 a) Compare the Runs A-D and Discuss your Observations! (3 Points)\n",
    "\n",
    "You must have noticed numerous differences between the 4 runs A-D. List at least 3 differences you have found. You may want to consider the elapsed time and the resulting association rules. Briefly explain your observations! For this subtask, you do not need to look at the movie names (`id_map=None`) as you observations are not specific to the context of movie recommendations; at this we will look in 2.2 b)\n",
    "\n",
    "**Your Answer:**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0cc66569",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "8d1bc62c",
   "metadata": {},
   "source": [
    "Now run the code cells above for Runs A-B again, but this time with `id_map=movie_map` so that the output will show for each rule the actual movie names.\n",
    "\n",
    "#### 2.2 b) Compare the Runs A-D and discuss the results for building a recommendation engine! (3 Points)\n",
    "\n",
    "Comparing the results of the different runs again, but now seeing the actual movie names, should give you some further insights how the choice of `min_support` and `min_confidence` might affect how the resulting rules are useful for building a recommendation engine.\n",
    "\n",
    "**Your Answer:**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d90e0931",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "64b84665",
   "metadata": {},
   "source": [
    "#### 2.2 c) Sketch a Movie Recommendation Algorithm Based on ARM (4 Points)\n",
    "\n",
    "So far, we only looked at individual rules and how the set of rules changes for different parameter values for `min_support` and `min_confidence`. However, we still need some method like `make_recommendation(shopping_history)` that takes the shopping history of a user and returns 1 or more recommendations. The goal is here is *not* to implement such a method but outline the main concerns to consider when implementing such a method\n",
    "\n",
    "\n",
    "(Hint: Do not forget that you not only have the information about Association Rules but also about the individual Frequent Itemsets)\n",
    "\n",
    "**Your Answer:**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7df19f8b",
   "metadata": {},
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.7.10 ('deeplearn_course')",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.10"
  },
  "vscode": {
   "interpreter": {
    "hash": "e22b1eee371346a4bb8d6d7d5605168edf2495926f53609f45bc0abb51b1607d"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
