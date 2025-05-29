import copy

import torch.nn as nn
import random

from local_differential_privacy_library import *
from util import *
from random import sample
from sklearn.metrics.pairwise import cosine_similarity

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)



class Client(nn.Module):
    def __init__(self, user_id, item_id, args):
        super().__init__()
        self.device = args.device
        self.user_id = user_id
        self.item_id = item_id #interacted items list
        #self.semantic_neighbors = semantic_neighbors


    def negative_sample(self, total_item_num):
        '''生成item负样本集合'''
        #从item列表里随机选取item作为user的负样本
        item_neg_ind = []
        #item_neg_ind和item_id数量一样
        for _ in self.item_id:
            neg_item = np.random.randint(1, total_item_num)
            while neg_item in self.item_id:
                neg_item = np.random.randint(1, total_item_num)
            item_neg_ind.append(neg_item)
        '''生成item负样本集合end'''
        return item_neg_ind

    def negative_sample_with_augment(self, total_item_num, sampled_items):
        item_set = self.item_id+sampled_items
        '''生成item负样本集合'''
        #从item列表里随机选取item作为user的负样本
        item_neg_ind = []
        #item_neg_ind和item_id数量一样
        for _ in item_set:
            neg_item = np.random.randint(1, total_item_num)
            while neg_item in item_set:
                neg_item = np.random.randint(1, total_item_num)
            item_neg_ind.append(neg_item)
        '''生成item负样本集合end'''
        return item_neg_ind

    def sample_item_augment(self, item_num):
        ls = [i for i in range(item_num) if i not in self.item_id]
        sampled_items = sample(ls, 5)

        return sampled_items


    def perturb_adj(self, value, label_author, author_label, label_count, shared_knowledge_rep, eps1, eps2):
        """
        *args* \n
        @**value**: i-th user's interaction vector 1*num_items in {0,1}\n
        @**label_author**: cluster-item mapping dict, indicating ids of items in each cluster, len(keys)=num_cluster\n
        @**author_label**: item-cluster mapping dict, indicating which cluster the item belongs to, len(keys)=num_item\n
        @**label_count**: the number of items in each cluster
        @**shared_knowledge_rep**: cluster representation\n
        @**eps1**: privacy budget for EM\n
        @**eps2**: privacy budget for RR\n
        *return*
        @**value**: perturbed interaction vector\n
        """
        # labels: author(item)属于哪一类，label_author:{cluster:author}, author_label:{author:cluster}
        # shared_knowledge_rep: 每个cluster的representation,1*14,该类a-c的平均
        #print(value.shape) #1,num_item
        #此用户的item共可分成多少个groups, 计算用户交互过的item共有几个group
        groups = {}
        for item in self.item_id:
            group = author_label[item]
            if(group not in groups.keys()):
                groups[group] = [item]
            else:
                groups[group].append(item)

        '''step1:EM'''
        '''计算每个cluster对于该user被选中的概率，如果user交互中出现过该cluster，其被选中概率就是1，否则则是计算该cluster与交互过的cluster的相似度最大值'''
        '''计算item分类的cluster对于该user的重要程度'''
        num_groups = len(groups)
        quality = np.array([0.0]*len(label_author)) # len=number of the total clusters
        G_s_u =  groups.keys() # 用户所交互过的item的cluster编号
        if(len(G_s_u)==0):#此用户没有交互的item，则各个位置quality平均
            for group in label_author.keys():
                quality[group] = 1
            num_groups = 1
        else:
            # 在cluster特征上（实际上是每个cluster在各个tag上的表现），对于每一个cluster，计算其在
            for group in label_author.keys():
                qua = max([(cosine_similarity(shared_knowledge_rep[g], shared_knowledge_rep[group])+1)/2.0 for g in G_s_u]) 
                # 如果group出现在了交互中，qua就是1，否则是交互中最相似的group（余弦相似度+1）/2
                quality[group] = qua

        EM_eps = eps1/num_groups
        EM_p = EM_eps*quality/2 #隐私预算1 eps
        EM_p = softmax(EM_p)

        #按照概率选择group
        select_group_keys = np.random.choice(range(len(label_author)), size = len(groups), replace = False, p = EM_p)
        select_group_keys_temp = list(select_group_keys)
        degree_list = [len(v) for _, v in groups.items()] # 保存每个cluster的交互数量
        new_groups = {}

        for key in select_group_keys:#先把存在于当前用户的shared knowledge拿出来
            key_temp = key
            if(key_temp in groups.keys()):
                new_groups[key_temp] = groups[key_temp]
                degree_list.remove(len(groups[key_temp]))
                select_group_keys_temp.remove(key_temp)

        for key in select_group_keys_temp:#不存在的随机采样交互的item，并保持度一致
            key_temp = key
            cur_degree = degree_list[0] # 直接使用剩下来的度
            if(len(label_author[key_temp]) >= cur_degree): # 可能存在所需要的度大于该cluster的item数量
                new_groups[key_temp] = random.sample(label_author[key_temp], cur_degree)
            else:#需要的度比当前group的size大，则将度设为当前group的size
                new_groups[key_temp] = label_author[key_temp]
            degree_list.remove(cur_degree)

        groups = new_groups # 重新构建稀疏交互向量
        value = np.zeros_like(value)#一定要更新value
        for group_id, items in groups.items():
            value[:,items] = 1
        '''pure em'''
        #value_rr = value



        '''step2:rr'''
        all_items = set(range(len(author_label)))
        select_items = []
        for group_id, items in groups.items():
            select_items.extend(label_author[group_id])
        mask_rr = list(all_items - set(select_items)) # 没有被选择的cluster中的所有item，这类item会被打上负标签，但由于选择是基于概率，原有cluster也有可能不被选中，造成原有的正item被打上硬负标签

        '''rr'''
        value_rr = perturbation_test(value, 1-value, eps2)
        #print(np.sum(value_rr)) 4648 513
        value_rr[:, mask_rr] = 0
        # #print(np.sum(value_rr)) 469 479
        #
        '''dprr'''
        for group_id, items in groups.items():
            degree = len(items)
            n = len(label_author[group_id])
            p = eps2p(eps2)
            q = degree/(degree*(2*p-1) + (n)*(1-p))
            rnd = np.random.random(value_rr.shape)
            #原来是0的一定还是0，原来是1的以概率q保持1，以达到degree减少
            dprr_results = np.where(rnd<q, value_rr, np.zeros((value_rr.shape))) 
            value_rr[:, label_author[group_id]] = dprr_results[:, label_author[group_id]] #  这两步确保只对group中的item进行操作


        print(f"....{self.user_id}....")
        print(f"True:\n{self.item_id}")
        print(f"Pertubated:\n{value_rr.nonzero()[1]}") ##### 这里会打印一串采样
        return value_rr





    def update(self, model_user, model_item):
        self.model_user = copy.deepcopy(model_user)
        self.model_item = copy.deepcopy(model_item)
        # self.item_emb.weight.data = Parameter(aggr_param['item'].weight.data.clone())


    def train_(self, hg, user_emb, item_emb):
        total_item_num = item_emb.weight.shape[0]
        user_emb = torch.clone(user_emb.weight).detach()
        item_emb = torch.clone(item_emb.weight).detach()
        user_emb.requires_grad = True
        item_emb.requires_grad = True
        user_emb.grad = torch.zeros_like(user_emb)
        item_emb.grad = torch.zeros_like(item_emb)


        self.model_user.train()
        self.model_item.train()

        #sample_item_augment
        sampled_item = self.sample_item_augment(total_item_num)
        item_neg_id = self.negative_sample_with_augment(total_item_num, sampled_item)
        #item_neg_id = self.negative_sample(total_item_num)

        logits_user = self.model_user(hg, user_emb)#+user_emb
        logits_item = self.model_item(hg, item_emb)#+item_emb

        cur_user = logits_user[self.user_id]
        #cur_item_pos = logits_item[self.item_id]
        cur_item_pos = logits_item[self.item_id+sampled_item]
        cur_item_neg = logits_item[item_neg_id]

        pos_scores = torch.sum(cur_user * cur_item_pos, dim=-1)
        neg_scores = torch.sum(cur_user * cur_item_neg, dim=-1)
        loss = -(pos_scores - neg_scores).sigmoid().log().sum()


        self.model_user.zero_grad()
        self.model_item.zero_grad()

        loss.backward()
        #self.optimizer.step()

        #grad
        model_grad_user = []
        model_grad_item = []
        for param in list(self.model_user.parameters()):
            grad = param.grad#
            model_grad_user.append(grad)
        for param in list(self.model_item.parameters()):
            grad = param.grad#
            model_grad_item.append(grad)

        mask_item = item_emb.grad.sum(-1)!=0#直接通过grad！=0
        updated_items = np.array(range(item_emb.shape[0]))[mask_item.cpu()]#list(set(self.item_id + item_neg_id))
        #print(updated_items)
        item_grad = item_emb.grad[updated_items, :]#


        mask_user = user_emb.grad.sum(-1)!=0
        updated_users = np.array(range(user_emb.shape[0]))[mask_user.cpu()]#list(set([self.user_id] + self.semantic_neighbors))
        #print(len(updated_users))
        user_grad = user_emb.grad[updated_users, :]#
        #print(user_grad)
        # torch.cuda.empty_cache()


        return {'user': (user_grad, updated_users), 'item' : (item_grad, updated_items), 'model': (model_grad_user, model_grad_item)}, \
               loss.detach()
