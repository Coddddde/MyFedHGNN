#coding:utf-8
import numpy as np
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.sparse
import logging

def print_log(file):
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,  # 设置日志级别，可以根据需要调整
        format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
        handlers=[
            logging.StreamHandler(),  # 输出到终端
            logging.FileHandler(file, mode='w'),  # 输出到文件
        ]
    )
    # 输出日志信息
    #logging.debug('信息将同时输出到终端和文件。')
    logging.info('信息会同时显示在终端和文件中。')


def pca_reduce(data, dim):
    pca = PCA(n_components=dim)
    pca = pca.fit(data)
    x = pca.transform(data)
    return x

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


'''cluster'''
def cluster(feature_list, n_clusters):
    s = KMeans(n_clusters=n_clusters).fit(np.asarray(feature_list))
    #print(len(s.cluster_centers_))
    #每个样本所属的簇
    #print(len(s.labels_))
    label_count = {}
    for i in s.labels_:
        if(i not in label_count.keys()):
            label_count[i] = 1
        else:
            label_count[i]+=1

    print(label_count)
    #print(s.labels_)

    label_author = {}
    author_label = {}
    labels = []
    for i, k in enumerate(s.labels_):
        author = i
        label = k
        labels.append(label)

        author_label[author] = label

        if(label not in label_author.keys()):
            label_author[label] = [author]
        else:
            label_author[label].append(author)

    # with open("./data_event/author_label", "w") as f:
    #     for l in author_label:
    #         f.write(l[0] + '\t' + l[1] + '\n')
    return label_count, labels, label_author, author_label
#cluster()

'''get shared knowledge rep(每个shared HIN的表示为它所包含的item的表示的平均）'''
def get_shared_knowledge_rep(item_feature_list, label_author):
    shared_knowledge_rep = {}
    for label, author_list in label_author.items():
        features = item_feature_list[author_list]
        rep = np.mean(features, 0)
        # sum = np.array([0.0]*len(item_feature_list[0]))
        # l = len(author_list)
        # for author in author_list:
        #     sum+= item_feature_list[author]
        # rep = sum/l
        shared_knowledge_rep[label] = np.asarray(rep)
    return shared_knowledge_rep


def tsne(feature_list):
    tsne = TSNE(n_components=2)
    tsne.fit_transform(feature_list)
    #print(tsne.embedding_)

    feature_list = tsne.embedding_
    print(np.shape(feature_list))#14795,2

    x = feature_list[:,0]
    y = feature_list[:,1]

    return x, y

#l = sio.loadmat("./data_event/author_tsne.mat")
# x= l['x']
# y =l['y']

'''
def plot_embedding_2d(x, y, labels):
    """Plot an embedding X with the class label y colored by the domain d."""
    # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)

    plt.scatter(x, y, c=labels)

    # plt.xlim((-1.5, 1.5))
    # plt.xticks([])  # ignore xticks
    # plt.ylim((-1.5, 1.5))
    # plt.yticks([])  # ignore yticks
    plt.show()
'''
#plot_embedding_2d(x,y)


def gen_shared_knowledge(adj, group_num):
    # p_vs_f = adj[0]#(4025,73)
    p_vs_a = adj[0]#(4025,17431)
    # p_vs_t = adj[2]#(4025,1903)
    p_vs_c = adj[1]#CSC (4025, 14)
    # a_vs_t = p_vs_a.T * p_vs_t
    # a_vs_f = p_vs_a.T * p_vs_f
    a_vs_c = p_vs_a.T * p_vs_c
    # a_vs_p = p_vs_a.T
    # a_vs_t_dense = a_vs_t.todense()
    # a_vs_f_dense = a_vs_f.todense()
    a_vs_c_dense = a_vs_c.todense()
    # a_vs_p_dense = a_vs_p.todense()
    #print(np.sum(a_vs_c_dense.sum(-1)==0))#大部分(10264)=0
    #print(a_vs_t_dense[1])
    a_feature = np.concatenate([a_vs_c_dense], -1) # 拿author-conference的关系，对author聚类
    label_count, labels, label_author, author_label = cluster(a_feature, group_num) #20
    # x,y = tsne(a_feature)
    # plot_embedding_2d(x, y, labels)
    shared_knowledge_rep = get_shared_knowledge_rep(a_feature, label_author)
    return label_count, labels, label_author, author_label, shared_knowledge_rep


# data loader for single domain test

def datasetFilter(ratings, min_items=5):
    """
            Only keep the data useful, which means:
                - all ratings are non-zeros
                - each user rated at least {self.min_items} items
            :param ratings: pd.DataFrame
            :param min_items: the least number of items user rated
            :return: filter_ratings: pd.DataFrame
            """

    # filter unuseful data
    ratings = ratings[ratings['rating'] > 0]

    # only keep users who rated at least {self.min_items} items
    user_count = ratings.groupby('uid').size()
    user_subset = np.in1d(ratings.uid, user_count[user_count >= min_items].index)
    filter_ratings = ratings[user_subset].reset_index(drop=True)

    del ratings

    return filter_ratings

def loadData(path, dataset, file_name='ratings.dat'):
    """
    @path: the path of the dataset
    @dataset: the name of the dataset
    @file_name: the name of the rating file
    """
    import os
    import pandas as pd

    dataset_file = os.path.join(path, dataset, file_name)

    min_rates = 10

    if dataset == "movielens":
        if file_name == 'ml-1m.dat':
            ratings = pd.read_csv(dataset_file, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                                engine='python')
        elif file_name == 'ml-100k.dat':
            ratings = pd.read_csv(dataset_file, sep='\t', header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                                engine='python')
        else:
            ratings = pd.DataFrame()
    elif dataset == "amazon":
        ratings = pd.read_csv(dataset_file, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                              engine='python')

    elif dataset == "books":

        min_rates = 5

        ratings = pd.read_csv(dataset_file, sep=",", header=1, usecols=[3, 4, 6], names=['uid', 'mid', 'rating'],
                              engine='python')

        # take the item orders instead of real timestamp
        rank = ratings[['mid']].drop_duplicates().reindex()
        rank['timestamp'] = np.arange((len(rank)))
        ratings = pd.merge(ratings, rank, on=['mid'], how='left')

    elif dataset == "last.fm":
        min_rates = 10

        ratings = pd.read_csv(dataset_file, sep="\t", header=0, usecols=[0, 1, 2], names=['uid', 'mid', 'rating'],
                              engine='python')

        # take the item orders instead of real timestamp
        rank = ratings[['mid']].drop_duplicates().reindex()
        rank['timestamp'] = np.arange((len(rank)))
        ratings = pd.merge(ratings, rank, on=['mid'], how='left')


    elif dataset == "user-behavior":
        chunks = pd.read_csv(dataset_file, sep=",", header=None, names=['uid', 'mid', 'cid', 'behavior', 'timestamp'],
                             engine='python', chunksize=1000000)

        all_chunks = []
        for chunk in chunks:
            chunk
            chunk.loc[chunk['behavior'] == 'pv', 'rating'] = 1
            chunk.loc[chunk['behavior'] == 'cart', 'rating'] = 2
            chunk.loc[chunk['behavior'] == 'fav', 'rating'] = 3
            chunk.loc[chunk['behavior'] == 'buy', 'rating'] = 4
            all_chunks.append(chunk)

        ratings = pd.concat(all_chunks)

    elif dataset == "tenrec":

        chunks = pd.read_csv(dataset_file, sep=",", header=1, usecols=[0, 1, 2],
                             names=['uid', 'mid', 'rating'],
                             engine='python', chunksize=1000000)

        all_chunks = []
        for chunk in chunks:
            all_chunks.append(chunk)

        ratings = pd.concat(all_chunks)

        # take the item orders instead of real timestamp
        rank = ratings[['mid']].drop_duplicates().reindex()
        rank['timestamp'] = np.arange((len(rank)))
        ratings = pd.merge(ratings, rank, on=['mid'], how='left')
    elif dataset == "douban":
        min_rates = 5
        ratings = pd.read_csv(dataset_file, sep="\t", header=None, names=['uid', 'mid', 'rating'], engine='python')
        rank = ratings[['mid']].drop_duplicates().reindex()
        rank['timestamp'] = np.arange((len(rank)))
        ratings = pd.merge(ratings, rank, on=['mid'], how='left')
    else:
        ratings = pd.DataFrame()

    ratings = datasetFilter(ratings, min_rates)

    # Reindex user id and item id
    user_id = ratings[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ratings = pd.merge(ratings, user_id, on=['uid'], how='left')

    item_id = ratings[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ratings = pd.merge(ratings, item_id, on=['mid'], how='left')

    ratings = ratings[['userId', 'itemId', 'rating', 'timestamp']].sort_values(by='userId', ascending=True)

    num_users, num_items = print_statistics(ratings)

    return ratings, num_users, num_items

def print_statistics(ratings):
    """print the statistics of the dataset, and return the number of users and items"""
    maxs = ratings.max()
    num_interactions = len(ratings)
    sparsity = 1 - num_interactions / ((maxs['userId'] + 1) * (maxs['itemId'] + 1))

    # logging.info('The number of users: {}, and of items: {}.'.format(int(maxs['userId'] + 1), int(maxs['itemId'] + 1)))
    # logging.info('There are total {} interactions, the sparsity is {:.2f}%.'.format(num_interactions, sparsity * 100))

    return int(maxs['userId'] + 1), int(maxs['itemId'] + 1)

# if __name__ == 'main':
#     # feature_list = []
#     # for index in author_id_list:#
#     #     fea = features[index]
#     #     #print(len(fea))
#     #     feature_list.append(fea)
#     # feature_list = np.array(feature_list)
