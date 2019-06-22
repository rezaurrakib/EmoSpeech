import os
import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ggplot import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def features_embedding_visualize_hidden_layer(dense_features, all_labels, path):
    os.chdir(path)
    vis_folder = ("FeatureEmbedding/")
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)

    transformed = TSNE(n_components=3, perplexity=50, n_iter=5000).fit_transform(dense_features)
    print("transformed.shape : ", transformed.shape)

    #colors = ['green', 'red', 'black', 'green', 'blue']
    #node_colors = []
    #for i in range(len(dense_features)):
    #    node_colors.append(colors[all_labels[i]])

    def label_printer(i):
        if i == 1:
            return "Class Neutral"
        elif i == 2:
            return "Class Angry"
        elif i == 3:
            return "Class Happy"
        else:
            return "Class Sad"

    df = pd.DataFrame()
    df['tsne_x'] = transformed[:, 0]
    df['tsne_y'] = transformed[:, 1]
    df['tsne_z'] = transformed[:, 2]
    df['label'] = all_labels
    df['label'] = df['label'].apply(label_printer)

    chart = ggplot(df, aes(x='tsne_x', y='tsne_y', z='tsne_z', color='label'))\
            + geom_point(size=70, alpha=0.9)\
            + ggtitle("Feature Embedding using SVM")

    chart.save(vis_folder + "feature_embd_test.png")

def features_embedding_visualize_3d(dense_features, all_labels, path):
    os.chdir(path)
    vis_folder = ("FeatureEmbedding/")
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)

    transformed = TSNE(n_components=3, perplexity=50, n_iter=5000).fit_transform(dense_features)
    #colors = ['r', 'g', '#0FDF9F', 'navy', 'b']
    #markers = ["s", "o", "D", "P", "4"]

    colors = ['r', '#f58231', 'g', '#ffe119', 'b', '#42d4f4', '#f032e6', '#e6beff']
    markers = ["s", "o", "D", "P", "4", "2", "*", "X"]
    node_colors = []
    node_markers = []
    for i in range(len(all_labels)):
        node_colors.append(colors[all_labels[i]])
        node_markers.append(markers[all_labels[i]])


    fig = plt.figure(figsize=(25, 17))
    ax = fig.add_subplot(111, projection='3d')
    X = transformed[:, 0]
    Y = transformed[:, 1]
    Z = transformed[:, 2]

    for x_point, y_point, z_point, color, m in zip(X, Y, Z, node_colors, node_markers):
        ax.scatter([x_point], [y_point], [z_point], s=30, c=color, marker=m)

    #ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], c=node_colors, marker=marker_printer)
    #ax.set_title("3D visualization of the Feature Embedding of Berlin Dataset")
    ax.set_title("3D visualization of the Feature Embedding of RAVDESS Dataset")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()

    plt.savefig("feture_embedding_ravdess_dataset_3d.png")
    plt.show()


def features_embedding_visualize_2d(dense_features, all_labels, path):
    os.chdir(path)
    vis_folder = ("FeatureEmbedding/")
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)

    transformed = TSNE(n_components=2, perplexity=50, n_iter=5000).fit_transform(dense_features)
    colors = ['r', 'g', '#0FDF9F', 'navy', 'b', '#5FDF9F', '#0F9F9F', '#0FD19F']
    markers = ["s", "o", "D", "P", "4", "2", "*", "X"]
    node_colors = []
    node_markers = []
    for i in range(len(all_labels)):
        node_colors.append(colors[all_labels[i]])
        node_markers.append(markers[all_labels[i]])

    fig = plt.figure(figsize=(25, 17))
    ax = fig.add_subplot(111, projection='rectilinear')
    X = transformed[:, 0]
    Y = transformed[:, 1]
    #Z = transformed[:, 2]

    for x_point, y_point, color, m in zip(X, Y, node_colors, node_markers):
        ax.scatter([x_point], [y_point], s=30, c=color, marker=m)

    # ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], c=node_colors, marker=marker_printer)
    ax.set_title("2D visualization of the Feature Embedding of Berlin Dataset")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')

    plt.savefig("feture_embedding_berlin_dataset_2d.png")
    plt.show()
