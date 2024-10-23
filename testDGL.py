import dgl
import torch as th
u, v = th.tensor([0, 1, 2]), th.tensor([2, 3, 4])
g = dgl.graph((u, v))
g.ndata['x'] = th.randn(5, 3)   # 原始特征在CPU上
print(g.device)
cuda_g = g.to('cuda:4')         # 接受来自后端框架的任何设备对象
cuda_g.device
cuda_g.ndata['x'].device        # 特征数据也拷贝到了GPU上
# 由GPU张量构造的图也在GPU上
u, v = u.to('cuda:4'), v.to('cuda:4')
g = dgl.graph((u, v))
print(g.device)