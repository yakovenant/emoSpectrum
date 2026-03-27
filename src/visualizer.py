import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils import custom_print


def plot_data_hist(data, title):

    data.plot(kind='bar', figsize=(8, 5)) # width=0.5, dpi=600
    plt.title(title)
    plt.xlabel('Emotion label')
    plt.ylabel('Number of samples')
    plt.xticks(rotation=0)
    plt.legend()
    plt.show()


def plot_loss_curves(train_loss_log, val_loss_log, epoch_best, epoch_stop, save_fig_to, logscale=True):

    fig, ax = plt.subplots()
    xaxis = np.linspace(1, len(train_loss_log), num=len(train_loss_log))
    ax.plot(xaxis, train_loss_log, marker='.', label="training loss")
    ax.plot(xaxis, val_loss_log, marker='.', label="validation loss")
    ax.axvline(x=epoch_best, ls="--", color='b', label='early stopping')
    '''if epoch_stop is not None:
        ax.axvline(x=epoch_stop, ls="--", color='r', label='confidence stop')'''
    ax.set_title("Loss curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(xaxis)
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    #ax.set_ylim(0, 1)
    ax.grid(True, which="both")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))
    fig.savefig(os.path.join(save_fig_to), dpi=600)
    plt.show()


def plot_curve(data_series, save_fig_to, title):

    fig, ax = plt.subplots()
    data_series = np.array(data_series)
    xaxis = np.linspace(1, len(data_series), num=len(data_series))
    if data_series.ndim == 1:
        ax.plot(xaxis, data_series, label=title) # marker='.'
    else:
        colors = plt.get_cmap('tab20').colors
        num_plots = np.shape(data_series)[-1]
        for i in range(0, num_plots):
            ax.plot(xaxis, data_series[:, i], label=str(i), color=colors[i]) # marker='.'
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_xticks(xaxis)
    ax.grid(True, which="both")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))
    fig.savefig(os.path.join(save_fig_to), dpi=600)
    plt.show()


@torch.no_grad()
def plot_embeddings_with_dataloader(model, dataloader, save_fig_to=None, embedding_reducer=TSNE(n_components=2, random_state=678), label_encoder=LabelEncoder()):

    def _eval_clusters(embs, labels):

        if isinstance(embs, np.ndarray): embs = torch.tensor(embs)
        if isinstance(labels, np.ndarray): labels = torch.tensor(labels)

        embs_norm = torch.nn.functional.normalize(embs, p=2, dim=1)
        unique_labels = torch.unique(labels)

        centroids = []
        intra_dists = []
        for cur_label in unique_labels:
            mask = (labels == cur_label)
            cur_class_embs = embs_norm[mask]

            cur_centroid = cur_class_embs.mean(dim=0, keepdim=True)
            cur_centroid = torch.nn.functional.normalize(cur_centroid, p=2, dim=1)
            centroids.append(cur_centroid)

            if len(cur_class_embs) > 1:
                cos_sim = torch.mm(cur_class_embs, cur_centroid.t())
                cur_intra = 1.0 - cos_sim.mean().item()
                intra_dists.append(max(0.0, cur_intra))
            else:
                intra_dists.append(0.0)

        intra_dists_avg = np.mean(intra_dists)
        centroids = torch.cat(centroids, dim=0)
        n_classes = len(unique_labels)

        if n_classes > 1:
            centroid_sim = torch.mm(centroids, centroids.t())
            triu_idxs = torch.triu_indices(n_classes, n_classes, offset=1)
            inter_dists_avg = (1.0 - centroid_sim[triu_idxs[0], triu_idxs[1]]).mean().item()
        else:
            inter_dists_avg = 1.0

        cluster_separation_ratio = inter_dists_avg / (inter_dists_avg + intra_dists_avg + 1e-16)

        custom_print(f"\nCluster Separation Ratio: {cluster_separation_ratio:.2f}")
        custom_print(f"\nAverage Intra-class Distance: {intra_dists_avg:.2f}\n")

        return cluster_separation_ratio, intra_dists_avg

    model.eval()
    plot_all_embeddings, plot_all_labels = [], []
    c = 0
    for batch in tqdm(dataloader, desc="Extract Embeddings"):
        plot_inputs = batch[0]
        plot_labels = batch[1]

        _, _, pooled_embedding = model(plot_inputs, return_embeddings=True, stat_pooling=model.params.stat_pooling)

        plot_all_embeddings.append(pooled_embedding.cpu().numpy())
        plot_all_labels.append(plot_labels.cpu().numpy())
        c += 1
    plot_all_embeddings = np.concatenate(plot_all_embeddings, axis=0)
    plot_all_labels = np.concatenate(plot_all_labels, axis=0)

    custom_print("\nPlot Embeddings...")
    plot_all_embeddings_2d = embedding_reducer.fit_transform(plot_all_embeddings)
    plot_all_labels_encoded = label_encoder.fit_transform(plot_all_labels)

    unique_labels = np.unique(plot_all_labels_encoded)
    fig = plt.figure(figsize=(8, 6)) # W, H, dpi=600
    for cur_label in unique_labels:
        idxs = (plot_all_labels_encoded == cur_label)
        plt.scatter(
            plot_all_embeddings_2d[idxs, 0],
            plot_all_embeddings_2d[idxs, 1],
            label=f'Emotion {cur_label}',
            #c=plot_all_labels_encoded,
            #cmap='gist_rainbow',
            marker='.')
    plt.grid(True)
    plt.legend()
    if save_fig_to is not None:
        fig.savefig(os.path.join(save_fig_to), dpi=600)
    else:
        fig.savefig(os.path.join(os.path.join(os.getcwd(), "embeddings.png")), dpi=600)
    plt.show(os.getcwd())

    custom_print("\nEvaluate clustering...")
    separation_ratio, intra_distance = _eval_clusters(plot_all_embeddings, plot_all_labels)

    return separation_ratio, intra_distance
