import neuralkg.model
from neuralkg.model.KGEModel.model import Model
from neuralkg.utils import setup_parser
from neuralkg.loss.Adv_Loss import Adv_Loss
import torch
from torch import nn
import torch.nn.functional as F
# neuralkg.model.RotatE

'''
修改源代码：
    parser部分修改，添加了alpha
    loss也要修改，添加上loss
'''

class RotPro(Model):

    '''
    Attribute:
        args: Model configuration parameters.
        epsilon: Calculate embedding_range.
        margin: Calculate embedding_range and loss. It is denoted by 'gamma' in origin paper.
        embedding_range: Uniform distribution range.
        ent_emb: Entity embedding, shape:[num_ent, emb_dim * 2].
        rel_emb: Relation_embedding, shape:[num_rel, emb_dim].
    '''

    def __init__(self, args):
        super(RotPro, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None

        self.init_emb()

    def init_emb(self):
        """Initialize the entity and relation embeddings in the form of a uniform distribution."""
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]),
            requires_grad=False
        )

        initializer = nn.init.uniform_

        # 构建实体和关系嵌入
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim * 2)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        initializer(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        initializer(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())

        # 构建投影矩阵所需的变量
        # 对角矩阵
        self.projection_a = nn.Parameter(torch.zeros([self.args.num_rel, self.args.emb_dim]))   # 对应于对角矩阵的a
        initializer(self.projection_a, a=0.5, b=0.5)
        self.projection_b = nn.Parameter(torch.zeros([self.args.num_rel, self.args.emb_dim]))   # 对应于对角矩阵的a
        initializer(self.projection_b, a=0.5, b=0.5)
        # 投影角θ_p 好像没有初始化
        self.projection_phase = nn.Parameter(torch.zeros([self.args.num_rel, self.args.emb_dim]))

    def score_func(self, head_emb, relation_emb, tail_emb, proj_a, proj_b, proj_theta, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation_emb / (self.embedding_range.item() / pi)

        # ---------------------------------projection---------------------------------
        re_projection = torch.cos(proj_theta)
        im_projection = torch.sin(proj_theta)
        '''
        | b*sin^2+a*cos^2  cos*sin*(b-a)   |
        | cos*sin*(b-a)    a*sin^2+b*cos^2 |

        | ma mb |
        | mb mc |
        '''
        ma = re_projection * re_projection * proj_a + im_projection * im_projection * proj_b
        mb = re_projection * im_projection * (proj_b - proj_a)
        md = re_projection * re_projection * proj_b + im_projection * im_projection * proj_a

        # 头实体投影结果
        re_head_proj = ma * re_head + mb * im_head
        im_head_proj = mb * re_head + md * im_head

        # 尾实体投影结果
        re_tail_proj = ma * re_tail + mb * im_tail
        im_tail_proj = mb * re_tail + md * im_tail

        # -----------------------------------rotate-----------------------------------
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        if mode == 'head-batch':
            re_score = re_relation * re_tail_proj + im_relation * im_tail_proj
            im_score = re_relation * im_tail_proj - im_relation * re_tail_proj
            re_score = re_score - re_head_proj
            im_score = im_score - im_head_proj
        else:
            re_score = re_head_proj * re_relation - im_head_proj * im_relation
            im_score = re_head_proj * im_relation + im_head_proj * re_relation
            re_score = re_score - re_tail_proj
            im_score = im_score - im_tail_proj

        # -----------------添加的部分------------------
        if self.args.use_improvement==1 and mode=='single':
            # print("proj_a:", proj_a.shape)
            # print("proj_b:", proj_b.shape)
            delta = torch.abs(proj_a - proj_b)
            # range = 1e-3
            # if delta - 0 <= range:
            #     # 此时delta->0  <=> a==b==1 不用算constrain， 因为不是那个关系
            #     constrain = 0
            # else:
            # arcsin  [-pi/2, pi/2]
            constrain = delta * torch.abs(
                torch.min(
                    torch.as_tensor([0.], device='cuda:0'),
                    torch.arctan(im_head_proj/re_head_proj)-torch.arctan(im_tail_proj/re_tail_proj)

                    # torch.arccos(re_head_proj/torch.sqrt(im_head_proj ** 2 + re_head_proj ** 2)) -
                    #     torch.arccos(re_tail_proj/torch.sqrt(im_tail_proj ** 2 + re_tail_proj ** 2))

                    # torch.arcsin(im_head_proj / torch.sqrt(im_head_proj ** 2 + re_head_proj ** 2)) -
                    # torch.arcsin(im_tail_proj / torch.sqrt(im_tail_proj ** 2 + re_tail_proj ** 2))
                )
            )
            # print("constrain:",constrain.shape) # []
        # norm
        # 各元素绝对值的平方求和开根号
        # -------------------------------------------

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        if mode == 'single' and self.args.use_improvement==1:
            print("constrain:",torch.max(constrain))
            # print("im_head:",torch.max(im_head_proj))
            # print("theta1:", torch.max(im_head_proj / torch.sqrt(im_head_proj ** 2 + re_head_proj ** 2)))
            # print("theta2:", torch.max(im_tail_proj / torch.sqrt(im_tail_proj ** 2 + re_tail_proj ** 2)))
            # print("score:",score)
            score += constrain
        # print("score:", score.shape)
        score = self.margin.item() - score.sum(dim=-1)
        return score

    def forward(self, triples, negs=None, mode='single'):
        '''

        :param triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
        :param negs: Negative samples, defaults to None.
        :param mode: Choose head-predict or tail-predict, Defaults to 'single'.
        :return:
        '''
        # print("111")
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        proj_a, proj_b, proj_theta = self.get_proj_param(triples, negs, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, proj_a, proj_b, proj_theta, mode)
        return score

    def get_proj_param(self, triples, negs, mode):
        # triple: [batch, 3]
        # negs: [batch, 256]
        # proj_a, proj_b, proj_theta = None, None, None
        # if mode == 'single':
        #     proj_a = self.projection_a(triples[:, 1]).unsqueeze(1)  # [batch, 1, dim]
        #     proj_b = self.projection_b(triples[:, 1]).unsqueeze(1)  # [batch, 1, dim]
        #     proj_theta = self.projection_phase(triples[:, 1]).unsqueeze(1)  # [batch, 1, dim]
        # elif mode == 'head-batch' or mode == 'head-predict':
        #     # 头部被替换
        #     # if negs == None: # 说明这个时候是在evluation，所以需要直接用所有的entity embedding
        #     proj_a = self.projection_a(triples[:,1]).unsqueeze(1)
        #     proj_b = self.projection_a(triples[:,1]).unsqueeze(1)
        #     proj_theta = self.projection_a(triples[:,1]).unsqueeze(1)
        # elif mode == 'tail-batch' or mode == 'tail-predict':
        #     pass
        if 'predict' in mode:
            # 应该不区分
            proj_a = self.projection_a[triples[:, 1]].unsqueeze(1)  # [batch, 1, dim]
            proj_b = self.projection_b[triples[:, 1]].unsqueeze(1)  # [batch, 1, dim]
            proj_theta = self.projection_phase[triples[:, 1]].unsqueeze(1)  # [batch, 1, dim]
        else:
            proj_a = self.projection_a[triples[:, 1]].unsqueeze(1)  # [batch, 1, dim]
            proj_b = self.projection_b[triples[:, 1]].unsqueeze(1)  # [batch, 1, dim]
            proj_theta = self.projection_phase[triples[:, 1]].unsqueeze(1)  # [batch, 1, dim]

        return proj_a, proj_b, proj_theta

    def get_score(self, batch, mode):
        """
        The functions used in the testing phase
        Args:
            batch: A batch of data.
            mode: Choose head-predict or tail-predict.
        Returns:
            score: The score of triples.
        """
        triples = batch["positive_sample"]
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        proj_a, proj_b, proj_theta = self.get_proj_param(triples, None, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, proj_a, proj_b, proj_theta, mode)
        return score

class PairRE(Model):
    def __init__(self, args):
        super(PairRE, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None

        self.init_emb()

    def init_emb(self):
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]),
            requires_grad=False
        )

        initializer = nn.init.uniform_

        # 构建实体和关系嵌入
        # 这里实体不用两倍，但是关系要为两倍
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim * 2)
        initializer(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        initializer(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        re_head, re_tail = torch.chunk(relation_emb, 2, -1)

        head = F.normalize(head_emb, 2, -1)
        tail = F.normalize(tail_emb, 2, -1)

        score = head * re_head - tail * re_tail
        score = self.margin.item() - torch.norm(score, p=1, dim=-1)
        return score


    def forward(self, triples, negs=None, mode='single'):
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        return score

    def get_score(self, batch, mode):
        triples = batch["positive_sample"]
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        return score

class RotProPair(Model):
    def __init__(self, args):
        super(RotProPair, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None

        self.init_emb()

    def init_emb(self):
        """Initialize the entity and relation embeddings in the form of a uniform distribution."""
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]),
            requires_grad=False
        )

        initializer = nn.init.uniform_

        # 构建实体和关系嵌入
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim * 2)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim * 4)           # 这边也要乘4, 0~1为r_H, 2~3为r_T, 其中每一部分是实部和虚部
        initializer(tensor=self.ent_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())
        initializer(tensor=self.rel_emb.weight.data, a=-self.embedding_range.item(), b=self.embedding_range.item())

        # 构建投影矩阵所需的变量
        # 对角矩阵
        self.projection_a = nn.Parameter(torch.zeros([self.args.num_rel, self.args.emb_dim]))  # 对应于对角矩阵的a
        initializer(self.projection_a, a=0.5, b=0.5)
        self.projection_b = nn.Parameter(torch.zeros([self.args.num_rel, self.args.emb_dim]))  # 对应于对角矩阵的a
        initializer(self.projection_b, a=0.5, b=0.5)
        # 投影角θ_p 好像没有初始化
        self.projection_phase = nn.Parameter(torch.zeros([self.args.num_rel, self.args.emb_dim]))

    def score_func(self, head_emb, relation_emb, tail_emb, proj_a, proj_b, proj_theta, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)

        # Make phases of relations uniformly distributed in [-pi, pi]
        # phase_relation = relation_emb / (self.embedding_range.item() / pi)

        # ---------------------------------projection---------------------------------
        re_projection = torch.cos(proj_theta)
        im_projection = torch.sin(proj_theta)
        '''
        | b*sin^2+a*cos^2  cos*sin*(b-a)   |
        | cos*sin*(b-a)    a*sin^2+b*cos^2 |

        | ma mb |
        | mb mc |
        '''
        ma = re_projection * re_projection * proj_a + im_projection * im_projection * proj_b
        mb = re_projection * im_projection * (proj_b - proj_a)
        md = re_projection * re_projection * proj_b + im_projection * im_projection * proj_a

        # 头实体投影结果
        re_head_proj = ma * re_head + mb * im_head
        im_head_proj = mb * re_head + md * im_head

        # 尾实体投影结果
        re_tail_proj = ma * re_tail + mb * im_tail
        im_tail_proj = mb * re_tail + md * im_tail

        # -----------------------------------rotate-----------------------------------

        re_rh, im_rh, re_rt, im_rt = torch.chunk(relation_emb, 4, dim=-1)
        re_score = re_head_proj * re_rh - im_head_proj * im_rh - (re_tail_proj * re_rt - im_tail_proj * im_rt)
        im_score = (re_head_proj * im_rh + im_head_proj * re_rh) - (re_tail_proj * im_rt + im_tail_proj * re_rt)
        # re_relation = torch.cos(phase_relation)
        # im_relation = torch.sin(phase_relation)
        # if mode == 'head-batch':
        #     re_score = re_relation * re_tail_proj + im_relation * im_tail_proj
        #     im_score = re_relation * im_tail_proj - im_relation * re_tail_proj
        #     re_score = re_score - re_head_proj
        #     im_score = im_score - im_head_proj
        # else:
        #     re_score = re_head_proj * re_relation - im_head_proj * im_relation
        #     im_score = re_head_proj * im_relation + im_head_proj * re_relation
        #     re_score = re_score - re_tail_proj
        #     im_score = im_score - im_tail_proj

        # -----------------添加的部分------------------
        if self.args.use_improvement == 1 and mode == 'single':
            # print("proj_a:", proj_a.shape)
            # print("proj_b:", proj_b.shape)
            delta = torch.abs(proj_a - proj_b)
            # range = 1e-3
            # if delta - 0 <= range:
            #     # 此时delta->0  <=> a==b==1 不用算constrain， 因为不是那个关系
            #     constrain = 0
            # else:
            # arcsin  [-pi/2, pi/2]
            constrain = delta * torch.abs(
                torch.min(
                    torch.as_tensor([0.], device='cuda:0'),
                    torch.arctan(im_head_proj / re_head_proj) - torch.arctan(im_tail_proj / re_tail_proj)

                    # torch.arccos(re_head_proj/torch.sqrt(im_head_proj ** 2 + re_head_proj ** 2)) -
                    #     torch.arccos(re_tail_proj/torch.sqrt(im_tail_proj ** 2 + re_tail_proj ** 2))

                    # torch.arcsin(im_head_proj / torch.sqrt(im_head_proj ** 2 + re_head_proj ** 2)) -
                    # torch.arcsin(im_tail_proj / torch.sqrt(im_tail_proj ** 2 + re_tail_proj ** 2))
                )
            )
            # print("constrain:",constrain.shape) # []
        # norm
        # 各元素绝对值的平方求和开根号
        # -------------------------------------------

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        if mode == 'single' and self.args.use_improvement == 1:
            print("constrain:", torch.max(constrain))
            # print("im_head:",torch.max(im_head_proj))
            # print("theta1:", torch.max(im_head_proj / torch.sqrt(im_head_proj ** 2 + re_head_proj ** 2)))
            # print("theta2:", torch.max(im_tail_proj / torch.sqrt(im_tail_proj ** 2 + re_tail_proj ** 2)))
            # print("score:",score)
            score += constrain
        # print("score:", score.shape)
        score = self.margin.item() - score.sum(dim=-1)
        return score

    def forward(self, triples, negs=None, mode='single'):
        '''

        :param triples: The triples ids, as (h, r, t), shape:[batch_size, 3].
        :param negs: Negative samples, defaults to None.
        :param mode: Choose head-predict or tail-predict, Defaults to 'single'.
        :return:
        '''
        # print("111")
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        proj_a, proj_b, proj_theta = self.get_proj_param(triples, negs, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, proj_a, proj_b, proj_theta, mode)
        return score

    def get_proj_param(self, triples, negs, mode):
        # triple: [batch, 3]
        # negs: [batch, 256]
        # proj_a, proj_b, proj_theta = None, None, None
        # if mode == 'single':
        #     proj_a = self.projection_a(triples[:, 1]).unsqueeze(1)  # [batch, 1, dim]
        #     proj_b = self.projection_b(triples[:, 1]).unsqueeze(1)  # [batch, 1, dim]
        #     proj_theta = self.projection_phase(triples[:, 1]).unsqueeze(1)  # [batch, 1, dim]
        # elif mode == 'head-batch' or mode == 'head-predict':
        #     # 头部被替换
        #     # if negs == None: # 说明这个时候是在evluation，所以需要直接用所有的entity embedding
        #     proj_a = self.projection_a(triples[:,1]).unsqueeze(1)
        #     proj_b = self.projection_a(triples[:,1]).unsqueeze(1)
        #     proj_theta = self.projection_a(triples[:,1]).unsqueeze(1)
        # elif mode == 'tail-batch' or mode == 'tail-predict':
        #     pass
        if 'predict' in mode:
            # 应该不区分
            proj_a = self.projection_a[triples[:, 1]].unsqueeze(1)  # [batch, 1, dim]
            proj_b = self.projection_b[triples[:, 1]].unsqueeze(1)  # [batch, 1, dim]
            proj_theta = self.projection_phase[triples[:, 1]].unsqueeze(1)  # [batch, 1, dim]
        else:
            proj_a = self.projection_a[triples[:, 1]].unsqueeze(1)  # [batch, 1, dim]
            proj_b = self.projection_b[triples[:, 1]].unsqueeze(1)  # [batch, 1, dim]
            proj_theta = self.projection_phase[triples[:, 1]].unsqueeze(1)  # [batch, 1, dim]

        return proj_a, proj_b, proj_theta

    def get_score(self, batch, mode):
        """
        The functions used in the testing phase
        Args:
            batch: A batch of data.
            mode: Choose head-predict or tail-predict.
        Returns:
            score: The score of triples.
        """
        triples = batch["positive_sample"]
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        proj_a, proj_b, proj_theta = self.get_proj_param(triples, None, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, proj_a, proj_b, proj_theta, mode)
        return score