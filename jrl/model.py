import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

from alias_multinomial import AliasMultinomial


def get_concat_embedding(example_idxs, embedding_list, embedding_masks=None):
    tmp_list = [None for i in range(len(embedding_list))]
    for i in range(len(embedding_list)):
        tmp_list[i] = embedding_list[i][example_idxs]
        if embedding_masks is not None:
            tmp_list[i] = tmp_list[i] * embedding_masks[i]
    return torch.cat(tmp_list, dim=1)


class Architecture:

    def __init__(self, model, cfg):
        self.cfg = cfg
        self.model = model
        self.arch_params = model.arch_parameters()
        self.optimizer = torch.optim.Adam(self.arch_params,
                                          cfg.MODEL.NAS.LR.MAX)
        #
        text, img = 0, 0
        if self.model.need_text:
            text = self.cfg.MODEL.EMBED_SIZE
        if self.model.need_image:
            img = self.cfg.MODEL.IMG_EMBED_SIZE

        abs_cost = text + img

        max_ratio = cfg.MODEL.NAS.BUDGET.MAX
        min_ratio = cfg.MODEL.NAS.BUDGET.MIN

        self.max_cost = abs_cost * max_ratio
        self.max_w = cfg.MODEL.NAS.BUDGET.MAX_W
        self.min_w = cfg.MODEL.NAS.BUDGET.MIN_W
        self.min_cost = abs_cost * min_ratio
        self.budget_scale = cfg.MODEL.NAS.BUDGET.SCALE
        self.budget = cfg.MODEL.NAS.BUDGET.ENABLED
        self.clip_norm = cfg.MODEL.NAS.GRAD_NORM
        self._loss, self._cost = 0, 0
        #
        self.invert_weight = 1 / cfg.MODEL.IMAGE_WEIGHT
        #self.distill = False
        if cfg.MODEL.NAS.DISTILL:
            self.distill = True
            self.model.enable_distillation(cfg.TRAIN.DISTILL.PATH)
            self.model.need_distill = False

    def __call__(self, model, input):
        obj_loss, partial_losses = model(input)

        # text_loss, image_loss = partial_losses[0:2]
        # if self.cfg.MODEL.NAS.INVERT_WEIGHT:
        #     image_loss = image_loss * self.invert_weight
        # self._loss = obj_loss = text_loss + image_loss
        self._loss = obj_loss
        self.optimizer.zero_grad()
        obj_loss.backward()
        nn.utils.clip_grad_norm_(model.arch_parameters(), self.clip_norm)
        self.optimizer.step()
        #

        self.model.reset_nas_weight()

    def decay(self):
        self.model.decay()

    def get_entropy(self):
        arch_state_dict = self.model.arch_state_dict()
        entropies = {}
        for k, v in arch_state_dict.items():
            prob = F.softmax(v, dim=-1)
            log_prob = F.log_softmax(v, dim=-1)
            entropies[k] = -(log_prob * prob).sum(-1,
                                                  keepdim=False).mean().item()
        return entropies

    def set_model_mask(self):
        if not os.path.isfile(self.cfg.TRAIN.MASK.PATH):
            print("architecture file does not exist. We do not load any mask")
            self.model.mode = 'train'
            return
        derivation = self.cfg.MODEL.NAS.DERIVATION
        assert derivation in ['selected', 'random', 'transference']
        nas_dict = torch.load(self.cfg.TRAIN.MASK.PATH)
        arch_state_dict = nas_dict['arch_state_dict']
        self.model.load_state_dict(arch_state_dict, strict=False)
        self.model.mode = 'train'
        if derivation == 'selected':
            # keep mask
            feature_embeddings = nas_dict['features']
            self.model.load_state_dict(feature_embeddings, strict=False)
            self.model.set_masks()
            self.model.init()
            self.model.load_state_dict(arch_state_dict, strict=False)
        elif derivation == 'random':
            self.model.set_masks(is_random=True)
        else:
            feature_embeddings = nas_dict['features']
            self.model.load_state_dict(feature_embeddings, strict=False)
            self.model.set_masks()

        self.model.reset_nas_weight()

    def state_dict(self):
        return self.optimizer.state_dict()

    def save_model_state(self):
        self._model_mode = self.model.mode
        self.model.mode = 'train'
        self.model.set_masks()
        self.model.reset_nas_weight()

    def resume_model_state(self):
        self.model.mode = self._model_mode

    def get_status(self):
        status = {}
        status['tau'] = self.model.tau
        status['valid loss'] = self._loss  # .detach().cpu().item()
        status['model cost'] = self._cost  # .detach().cpu().item()
        status['arch lr'] = self.optimizer.param_groups[0]['lr']
        status['sdl'] = self.model.sdl
        status.update(self.get_entropy())
        status.update(self.get_num_features())
        status['combine_func'] = self.model.combine_func[
            self.model.arch_combine.argmax()]
        return status

    def get_num_features(self):
        status = {}
        arch_status = self.model.decode_arch(mode='train')

        status['num-text-features'] = arch_status.get('num_text_features', 0)
        status['num-image-features'] = arch_status.get('num_image_features', 0)

        status['num-text-features-sampled'] = getattr(self.model,
                                                      'text_num_features', 0)
        status['num-image-features-sampled'] = getattr(self.model,
                                                       'image_num_features', 0)

        return status

    def get_budget(self):
        # num_features = self.get_num_features()
        text, image = 0, 0
        #
        if self.model.need_text:
            text = self.model.text_num_features * self.model.text_nas_weight  # .num_features['num-text-features']
        if self.model.need_image:
            image = self.model.image_num_features * self.model.image_nas_weight  # num_features['num-image-features']

        cost = text + image
        loss = max(0, cost / self.max_cost - 1) * self.budget_scale
        # loss = (cost / self.max_cost)**(0.07) * self.budget_scale
        # loss = (max(0, cost - self.max_cost) * self.max_w + max(0, self.max_cost - cost) * self.min_w) * self.budget_scale
        return loss

    def modalities_loss(self, inputs):
        u_idx, p_idx = inputs[0][:, 0], inputs[0][:, 1]
        loss_text = loss_image = loss_joint = None

        if self.model.need_text:
            loss_text = self.modality_loss(self.model.user_emb,
                                           self.model.product_emb,
                                           self.model.teacher_user_emb,
                                           self.model.teacher_product_emb,
                                           self.model.user_emb_mask, u_idx,
                                           p_idx)
        if self.model.need_image:
            loss_image = self.modality_loss(self.model.img_user_emb,
                                            self.model.img_product_emb,
                                            self.model.teacher_img_user_emb,
                                            self.model.teacher_img_product_emb,
                                            self.model.img_user_emb_mask,
                                            u_idx, p_idx)

        if self.model.need_text and self.model.need_image:
            u = torch.cat([self.model.user_emb, self.model.img_user_emb],
                          dim=-1)
            p = torch.cat([self.model.product_emb, self.model.img_product_emb],
                          dim=-1)
            m = torch.cat(
                [self.model.user_emb_mask, self.model.img_user_emb_mask],
                dim=-1)
            tu = torch.cat(
                [self.model.teacher_user_emb, self.model.teacher_img_user_emb],
                dim=-1)
            tp = torch.cat([
                self.model.teacher_product_emb,
                self.model.teacher_img_product_emb
            ],
                           dim=-1)

            loss_joint = self.modality_loss(u, p, tu, tp, m, u_idx, p_idx)

        total_loss = (loss_text + loss_image + loss_joint) / 3
        return total_loss * self.model.text_nas_weight

    def modality_loss(self, u, p, tu, tp, m, u_idx, p_idx):
        emb_u = u[u_idx] * m
        emb_p = p[p_idx] * m
        emb_tu = tu[u_idx]
        emb_tp = tp[p_idx]
        return self.model.distill_loss(emb_u, emb_p, emb_tu, emb_tp)


class MultiViewEmbeddingModel(nn.Module):

    def __init__(self, cfg: dict, sizes: dict, distributions: dict):
        super().__init__()
        self.cfg = cfg
        self.sizes = sizes
        self.distributions = distributions
        self.img_weight = cfg.MODEL.IMAGE_WEIGHT
        self.negative_samples = cfg.MODEL.NEGATIVE_SAMPLES
        self.settings = settings = cfg.MODEL
        self.similarity_func = cfg.MODEL.SIMILARITY_FUNC
        self.rank_cutoff = cfg.MODEL.RANK_CUTOFF
        self.init_width = 0.5 / settings.EMBED_SIZE
        self.need_distill = False
        self.embed_size = settings.EMBED_SIZE
        self.need_text = False
        self.need_image = False
        self.need_mask = False
        self.need_review = settings.NEED_REVIEW
        ### nas options
        self.feature_steps = cfg.MODEL.NAS.STEPS
        self.selection_method = cfg.MODEL.NAS.SELECTION
        assert self.selection_method in ['selection', 'cutoff', 'unified']
        self.mode = 'search'  # there will be two mode "search" and "train"
        self.tau_max, self.tau_min = cfg.MODEL.NAS.TAU
        self.tau_decay = (self.tau_max - self.tau_min) / (
            cfg.TRAIN.END_EPOCH - cfg.TRAIN.BEGIN_EPOCH)
        self.tau = self.tau_max
        self.image_nas_weight = 1
        self.text_nas_weight = 1
        self.need_self_distillation = cfg.MODEL.NAS.SELF_DISTILL

        ###

        if settings.NEED_TEXT:
            self.need_text = True
            self.register("word_emb",
                          torch.empty(sizes.vocab_size, self.embed_size))
            self.register("word_bias", torch.zeros(sizes.vocab_size))
            self.register("user_emb", torch.empty(sizes.user, self.embed_size))
            self.register("product_emb",
                          torch.empty(sizes.product, self.embed_size))
            self.register("product_bias", torch.zeros(sizes.product))

            if settings.NEED_REVIEW:
                self.register("review_emb",
                              torch.empty(sizes.review, self.embed_size))
                self.register("review_bias", torch.empty(sizes.review))

            self.need_mask = True
            load_as = 'buffer'  # if cfg.TRAIN.MASK.BUFFER else 'parameter'
            init1 = torch.ones(self.embed_size)
            init2 = torch.ones(self.embed_size)
            self.register("user_emb_mask", init1, load_as)
            self.register("product_emb_mask", init2, load_as)
            offset = self.embed_size // self.feature_steps
            self.text_feature_sizes = [
                self.embed_size - offset * i for i in range(self.feature_steps)
            ]
            self.text_num_features = settings.EMBED_SIZE
            # self.register("arch_text", torch.empty(len(self.text_feature_sizes)))

        if settings.NEED_IMAGE:
            self.need_image = True
            self.register("img_user_emb",
                          torch.empty(sizes.user, settings.IMG_EMBED_SIZE))
            self.register("img_product_emb",
                          torch.empty(sizes.product, settings.IMG_EMBED_SIZE))
            self.register("img_product_bias", torch.zeros(sizes.product))

            self.need_mask = True
            load_as = 'buffer'  # if cfg.TRAIN.MASK.BUFFER else 'parameter'
            init1 = torch.ones(settings.IMG_EMBED_SIZE)
            init2 = torch.ones(settings.IMG_EMBED_SIZE)

            self.register("img_user_emb_mask", init1, load_as)
            self.register("img_product_emb_mask", init2, load_as)

            self.user_linear = nn.Sequential(*[
                nn.Linear(self.cfg.MODEL.IMG_EMBED_SIZE, 4096 // 2),
                nn.ELU(),
                nn.Linear(4096 // 2, 4096),
                nn.ELU()
            ])
            self.product_linear = nn.Sequential(*[
                nn.Linear(self.cfg.MODEL.IMG_EMBED_SIZE, 4096 // 2),
                nn.ELU(),
                nn.Linear(4096 // 2, 4096),
                nn.ELU()
            ])
            offset = settings.IMG_EMBED_SIZE // self.feature_steps
            self.image_feature_sizes = [
                self.embed_size - offset * i for i in range(self.feature_steps)
            ]
            self.image_num_features = settings.IMG_EMBED_SIZE

        self.need_bpr = False
        if settings.NEED_BPR:
            self.need_bpr = True
            self.register("overall_product_bias", torch.zeros(sizes.product))

        self.alias_words = AliasMultinomial(self.distributions.words)
        self.alias_reviews = AliasMultinomial(self.distributions.reviews)
        self.alias_products = AliasMultinomial(self.distributions.products)

        total_features = 0

        if self.need_text:
            total_features += settings.EMBED_SIZE
        if self.need_image:
            total_features += settings.IMG_EMBED_SIZE
        max_features = int(total_features * settings.NAS.BUDGET.MAX)
        min_features = int(total_features * settings.NAS.BUDGET.MIN)

        self.feature_sizes = []
        if self.need_text and self.need_image:
            for t in self.text_feature_sizes:
                for i in self.image_feature_sizes:
                    _sum = t + i
                    if _sum <= max_features and _sum >= min_features:
                        self.feature_sizes.append([t, i])
        else:
            for i in range(min_features, max_features + 1, 1):
                self.feature_sizes.append([i])
        self.register("arch", torch.empty(len(self.feature_sizes)))

        self.combine_func = cfg.MODEL.NAS.COMBINE_FUNC
        self.register("arch_combine", torch.empty(len(self.combine_func)))
        self.init()

        print("architecture size is: ", len(self.feature_sizes))
        self.sdl = 0.0
        self._op_w = [0 for i in range(len(self.combine_func))]
        self._op_w[0] = 1

    def reset_nas_weight(self):
        if self.need_image:
            self.img_user_emb_mask = self.img_user_emb_mask.detach().clone(
            )  # .data.copy_(u_m.data)
            self.img_product_emb_mask = self.img_product_emb_mask.detach(
            ).clone()
        if self.need_text:
            self.user_emb_mask = self.user_emb_mask.detach().clone()
            self.product_emb_mask = self.product_emb_mask.detach().clone()

    def net_parameters(self):
        for name, param in self.named_parameters():
            if 'arch' in name: continue
            yield param

    def arch_parameters(self):
        return [self.arch, self.arch_combine]

    def arch_state_dict(self):
        return {'arch': self.arch, 'arch_combine': self.arch_combine}

    def feature_state_dict(self):
        state_dict = {}
        if self.need_text:
            state_dict['user_emb'] = self.user_emb
            state_dict['product_emb'] = self.product_emb
        if self.need_image:
            state_dict['img_user_emb'] = self.img_user_emb
            state_dict['img_product_emb'] = self.img_product_emb
        return state_dict

    def decay(self):
        self.tau = max(self.tau - self.tau_decay, self.tau_min)

    def softmax(self, mode, arch):
        if mode == 'search':
            soft_arch = F.gumbel_softmax(F.log_softmax(arch, dim=-1),
                                         tau=self.tau,
                                         hard=True,
                                         dim=-1)
        else:
            max_idx = torch.argmax(arch, -1, keepdim=True)
            one_hot = torch.zeros_like(arch)
            one_hot.scatter_(-1, max_idx, 1)  # onehot but detach from graph
            soft_arch = one_hot - arch.detach() + arch  # attach to gradient
        return soft_arch

    def decode_arch(self, mode):
        ret = {}
        resource = self.softmax(mode, self.arch)
        index = resource.argmax()
        weight = resource[index]
        nums = self.feature_sizes[index]
        if self.need_text:
            ret['num_text_features'] = nums[
                0]  # self.text_feature_sizes[index % len(self.text_feature_sizes)]
        if self.need_image:
            ret['num_image_features'] = nums[-1]
        ret['weight'] = weight
        #
        operators = self.softmax(mode, self.arch_combine)
        ret['op_w'] = operators
        #
        return ret

    def set_masks(self, is_random=False):
        values = self.decode_arch(self.mode)
        if self.need_image:
            u_m, p_m, w, num_features = self._set_mask(
                values['num_image_features'], values['weight'],
                self.img_user_emb, self.img_product_emb, is_random)
            self.img_user_emb_mask = u_m
            self.img_product_emb_mask = p_m
            self.image_nas_weight = w
            self.image_num_features = num_features
        if self.need_text:
            u_m, p_m, w, num_features = self._set_mask(
                values['num_text_features'], values['weight'], self.user_emb,
                self.product_emb, is_random)
            self.user_emb_mask = u_m  # data.copy_(u_m.data)
            self.product_emb_mask = p_m  # data.copy_(p_m.data)
            self.text_nas_weight = w
            self.text_num_features = num_features

        self._op_w = values['op_w']

    def _set_mask(self, num_features, weight, u, p, is_random=False):
        user_mask = torch.ones_like(u[0])
        product_mask = torch.ones_like(p[0])
        if not is_random:
            if self.selection_method == 'cutoff':
                user_mask[num_features::] = 0
                product_mask[num_features::] = 0
            elif self.selection_method == 'unified':
                avg_user = LA.norm(u, dim=0).unsqueeze(
                    0)  # u.sum(dim=0).div(u.shape[0])
                avg_product = LA.norm(p, dim=0).unsqueeze(
                    0)  # p.sum(dim=0).div(p.shape[0])

                avg_u_p = avg_user * avg_product

                _, top_index = avg_u_p.topk(avg_u_p.shape[-1])
                top_index = top_index[0]
                user_mask[:] = 0
                product_mask[:] = 0
                user_mask[top_index[0:num_features]] = 1
                product_mask[top_index[0:num_features]] = 1

                assert user_mask.sum(dim=-1).squeeze() == num_features

            else:
                avg_user = u.sum(dim=0).div(u.shape[0])
                avg_product = p.sum(dim=0).div(p.shape[0])

                _, top_index_u = avg_user.topk(avg_user.shape[-1])
                _, top_index_p = avg_product.topk(avg_product.shape[-1])

                user_mask[:] = 0
                product_mask[:] = 0
                user_mask[top_index_u[0:num_features]] = 1
                product_mask[top_index_p[0:num_features]] = 1

        else:
            idx = [i for i in range(u.shape[-1])]
            random_index = random.choices(idx, k=num_features)
            user_mask[:] = 0
            product_mask[:] = 0
            user_mask[random_index] = 1
            product_mask[random_index] = 1
        return user_mask * weight, product_mask * weight, weight, num_features

    def disable_nas(self):
        self.mode = 'train'

    def init(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'mask' in name:
                torch.nn.init.constant_(param, 1.0)
            elif 'word' in name:
                torch.nn.init.uniform_(param, -self.init_width,
                                       self.init_width)
            elif 'arch' in name:
                torch.nn.init.constant_(param, 1e-3)
            elif 'linear' in name:
                print(f"init linear layer: {name}")
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.constant_(param, 1e-8)

    def enable_distillation(self, path: str):
        self.need_distill = True
        assert os.path.isfile(path)
        distill_model = torch.load(path, map_location='cpu')
        for k, v in distill_model.items():
            if "user" in k or "product" in k:
                if 'alias' not in k and 'weight' not in k and 'bias' not in k:
                    print(f"registering {k} from {path}")
                    self.register_buffer(f"teacher_{k}", v)
        if hasattr(self, f"teacher_user_emb_mask"):
            mask_user = self.teacher_user_emb * self.teacher_user_emb_mask
            mask_product = self.teacher_product_emb * self.teacher_product_emb_mask
            self.teacher_user_emb.data.copy_(mask_user.data)
            self.teacher_product_emb.data.copy_(mask_product.data)
        if hasattr(self, 'teacher_img_user_emb_mask'):
            mask_user = self.teacher_img_user_emb * self.teacher_img_user_emb_mask
            mask_product = self.teacher_img_product_emb * self.teacher_img_product_emb_mask
            self.teacher_img_user_emb.data.copy_(mask_user.data)
            self.teacher_img_product_emb.data.copy_(mask_product.data)

    def register(self, name: str, data, load_as='parameter'):
        # dtype = torch.float32
        # if self.cfg.MODEL.QUANTIZE:
        # pass
        # # data.type(torch.qint8)
        eval(f'self.register_{load_as}')(name, torch.nn.Parameter(data))
        # self.register_parameter(name, torch.nn.Parameter(data))

    def forward(self, inputs):
        if self.training:
            if self.mode != 'train':
                self.set_masks()
            return self.forward_train(inputs)
        else:
            return self.get_product_scores(*inputs)

    def forward_train(self, inputs):
        assert len(inputs) == 2
        user_idxs , product_idxs , review_idxs , word_idxs = inputs[
                                                                 0][: , 0] , inputs[0][: , 1] , inputs[0][: , 2] , \
                                                             inputs[0][: , 3]
        img_fea = inputs[1]
        losses, l2_losses = [], []

        if self.need_text:
            review_loss, l2_loss = self.review_nce_loss(
                user_idxs, product_idxs, review_idxs, word_idxs)
            losses.append(review_loss)
            l2_losses += l2_loss

            if self.settings.NEED_BPR and self.settings.NEED_EXTEND:
                r_bpr_loss, regularize_emb_list = self.single_nce_loss(
                    user_idxs, self.user_emb, product_idxs, self.product_emb,
                    self.product_bias, self.alias_products, self.user_emb_mask,
                    self.product_emb_mask)
                losses.append(r_bpr_loss)
                l2_losses += regularize_emb_list

        if self.need_image:
            image_loss, regularize_emb_list = self.img_nce_loss(
                user_idxs, product_idxs, img_fea)
            losses.append(image_loss * self.img_weight)
            l2_losses += regularize_emb_list

            if self.settings.NEED_BPR and self.settings.NEED_EXTEND:
                img_bpr_loss, regularize_emb_list = self.single_nce_loss(
                    user_idxs, self.img_user_emb, product_idxs,
                    self.img_product_emb, self.img_product_bias,
                    self.alias_products, self.img_user_emb_mask,
                    self.img_product_emb_mask)
                losses.append(img_bpr_loss)
                l2_losses.append(regularize_emb_list)

        if self.need_bpr and not self.settings.NEED_EXTEND:
            bpr_loss, regularize_emb_list = self.multiview_bpr_loss(
                user_idxs, product_idxs)
            # bpr_loss *= self._op_w[self._op_w.argmax()]
            losses.append(bpr_loss)
            l2_losses.append(regularize_emb_list)

        if self.need_distill:
            loss_distill = self.get_distill_loss(user_idxs, product_idxs)
            losses.append(loss_distill)
        elif self.need_self_distillation:
            loss_distill = self.get_self_distill_loss(user_idxs, product_idxs)
            losses.append(loss_distill)

        overall_loss = sum(losses)
        return overall_loss, losses

    def get_distill_loss(self, uidxs, pidxs):
        loss = 0
        user, product, _ = self.get_all_user_product_embedding_list()
        user_mask, product_mask = self.get_all_embeddings_masks()

        u = self.get_combined_features(uidxs, user, user_mask)
        p = self.get_combined_features(pidxs, product, product_mask)

        tu = get_concat_embedding(
            uidxs, [self.teacher_user_emb, self.teacher_img_user_emb])
        tp = get_concat_embedding(
            pidxs, [self.teacher_product_emb, self.teacher_img_product_emb])

        loss = self.distill_loss(u, p, tu, tp)
        self.sdl = loss * self.cfg.TRAIN.DISTILL.SCALE
        return loss * self.cfg.TRAIN.DISTILL.SCALE

    def get_self_distill_loss(self, uidxs, pidxs):

        loss = 0
        if self.need_text:
            user_emb = self.get_embeddings(self.user_emb, uidxs,
                                           self.user_emb_mask)
            product_emb = self.get_embeddings(self.product_emb, pidxs,
                                              self.product_emb_mask)

            student_user_product_adj = torch.matmul(user_emb, product_emb.t())

            teacher_user_product_adj = torch.matmul(
                self.user_emb[uidxs].detach().clone(),
                self.product_emb[pidxs].detach().clone().t())
            loss += F.kl_div(F.log_softmax(student_user_product_adj, dim=-1),
                             F.softmax(teacher_user_product_adj, dim=-1))
            loss += F.kl_div(
                F.log_softmax(student_user_product_adj.t(), dim=-1),
                F.softmax(teacher_user_product_adj.t(), dim=-1))
        if self.need_image:
            img_user_emb = self.get_embeddings(self.img_user_emb, uidxs,
                                               self.img_user_emb_mask)
            img_product_emb = self.get_embeddings(self.img_product_emb, pidxs,
                                                  self.img_product_emb_mask)
            student_user_product_adj = torch.matmul(img_user_emb,
                                                    img_product_emb.t())
            teacher_user_product_adj = torch.matmul(
                self.img_user_emb[uidxs].detach().clone(),
                self.img_product_emb[pidxs].detach().clone().t())
            loss += F.kl_div(F.log_softmax(student_user_product_adj, dim=-1),
                             F.softmax(teacher_user_product_adj, dim=-1))
            loss += F.kl_div(
                F.log_softmax(student_user_product_adj.t(), dim=-1),
                F.softmax(teacher_user_product_adj.t(), dim=-1))
        self.sdl = loss * self.cfg.TRAIN.DISTILL.SCALE
        return loss * self.cfg.TRAIN.DISTILL.SCALE

    def distill_loss(self, u, p, tu, tp):
        upa = torch.matmul(u, p.t())
        tupa = torch.matmul(tu, tp.t())
        loss = F.kl_div(F.log_softmax(upa, dim=-1),
                        F.softmax(tupa, dim=-1),
                        reduction='batchmean')

        loss_inv = F.kl_div(F.log_softmax(upa.t(), dim=-1),
                            F.softmax(tupa.t(), dim=-1),
                            reduction='batchmean')
        return loss + loss_inv

    def multiview_bpr_loss(self, user_idxs, product_idxs):

        # neg sampling
        neg_product_idxs = self.alias_products.draw(self.negative_samples)

        user_emb_list, product_emb_list, _ = self.get_all_user_product_embedding_list(
        )
        user_emb_masks, product_emb_masks = self.get_all_embeddings_masks()

        # concat user embeddings
        example_vec = self.get_combined_features(user_idxs, user_emb_list,
                                                 user_emb_masks)
        true_w = self.get_combined_features(product_idxs, product_emb_list,
                                            product_emb_masks)
        true_b = self.overall_product_bias[product_idxs]

        # get sampled embeddings and bias [num_sampled, embed_size], [num_sampled, 1]
        sampled_w = self.get_combined_features(neg_product_idxs,
                                               product_emb_list,
                                               product_emb_masks)
        sampled_b = self.overall_product_bias[neg_product_idxs]

        # True logits: [batch_size, 1]
        true_logit = torch.sum(example_vec.mul(true_w), dim=-1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise lables for all examples in the batch
        # using the matmul.
        sampled_logit = torch.matmul(example_vec, sampled_w.transpose(
            0, 1)) + sampled_b

        loss = self.nce_loss(true_logit,
                             sampled_logit), [example_vec, true_w, sampled_w]
        return loss

    def get_all_embeddings_masks(self, ):
        user_masks = []
        product_masks = []
        if self.need_text:
            user_masks.append(self.user_emb_mask * self.text_nas_weight)
            product_masks.append(self.product_emb_mask * self.text_nas_weight)
        if self.need_image:
            user_masks.append(self.img_user_emb_mask * self.image_nas_weight)
            product_masks.append(self.img_product_emb_mask *
                                 self.image_nas_weight)
        return user_masks, product_masks

    def get_all_user_product_embedding_list(self):
        user_emb_list = []
        product_emb_list = []
        product_bias_list = []
        if self.need_text:
            user_emb_list.append(self.user_emb)
            product_emb_list.append(self.product_emb)
            product_bias_list.append(self.product_bias)
        if self.need_image:
            user_emb_list.append(self.img_user_emb)
            product_emb_list.append(self.img_product_emb)
            product_bias_list.append(self.img_product_bias)
        return user_emb_list, product_emb_list, product_bias_list

    def similarity_function(self, user_vec, product_vec, product_bias):
        if self.similarity_func == 'product':
            return torch.matmul(user_vec, product_vec.transpose(0, 1))
        elif self.similarity_func == 'bias_product':
            return torch.matmul(user_vec, product_vec.transpose(
                0, 1)) + product_bias
        else:
            user_norm = torch.norm(
                user_vec, keep_dims=True
            )  # torch.sqrt(torch.sum(user_vec**2, 1, keep_dims=True))
            product_norm = torch.norm(product_vec, keep_dims=True)
            ratio_user = user_vec / user_norm
            ratio_prod = product_vec / product_norm
            return torch.matmul(ratio_user, ratio_prod.transpose(0, 1))

    def get_product_scores(self, user_idxs, product_idxs=None):
        # get needed embeddings
        user_emb_list, product_emb_list, product_bias_list = self.get_all_user_product_embedding_list(
        )
        user_masks, product_masks = self.get_all_embeddings_masks()
        # get user embedding [None, n*embed_size]
        user_vec = self.get_combined_features(user_idxs, user_emb_list,
                                              user_masks)
        # get candidate product embedding [None, embed_size]
        product_vec = None
        product_bias = None
        # if product_idxs != None:
        if product_idxs == None:
            product_idxs = [i for i in range(product_emb_list[0].size(0))]
        product_vec = self.get_combined_features(product_idxs,
                                                 product_emb_list,
                                                 product_masks)
        product_bias = self.overall_product_bias[product_idxs]
        # else:
        #     product_vec = torch.cat(product_emb_list , dim=1)
        #     product_bias = self.overall_product_bias
        return self.similarity_function(user_vec, product_vec, product_bias)

    def img_nce_loss(self, uidxs, pidxs, img_vec):
        user_vec = self.get_embeddings(
            self.img_user_emb, uidxs, self.img_user_emb_mask,
            self.image_nas_weight)  # self.img_user_emb[uidxs]
        product_vec = self.get_embeddings(
            self.img_product_emb, pidxs, self.img_product_emb_mask,
            self.image_nas_weight)  # self.img_product_emb[pidxs]

        user_img_vec = self.user_linear(user_vec)  # * self.image_nas_weight
        product_img_vec = self.product_linear(
            product_vec)  # * self.image_nas_weight

        loss = F.mse_loss(user_img_vec, img_vec)
        loss += F.mse_loss(product_img_vec, img_vec)

        return loss, [user_vec, product_vec]

    def decode_img(self, input_data):
        output_data = input_data
        output_sizes = [int((4096 + self.cfg.MODEL.IMG_EMBED_SIZE) / 2), 4096]
        current_size = self.cfg.MODEL.IMG_EMBED_SIZE
        for i in range(len(output_sizes)):
            name_w = f"expand_W_{current_size}_{output_sizes[i]}"
            name_b = f"expand_b_{output_sizes[i]}"
            w = getattr(self, name_w)
            b = getattr(self, name_b)
            output_data = torch.matmul(output_data, w) + b
            output_data = F.elu(output_data)
            current_size = output_sizes[i]
        return output_data

    def review_nce_loss(self, uidxs, pidxs, ridxs, widxs):
        loss_list, l2_loss_list = [], []
        if self.settings.NEED_REVIEW:
            ur_loss, ur_emb_list = self.single_nce_loss(
                uidxs,
                self.user_emb,
                ridxs,
                self.review_emb,
                self.review_bias,
                self.alias_reviews,
                example_mask=self.user_emb_mask,
                example_weight=self.text_nas_weight)
            pr_loss, pr_emb_list = self.single_nce_loss(
                pidxs,
                self.product_emb,
                ridxs,
                self.review_emb,
                self.review_bias,
                self.alias_reviews,
                example_mask=self.product_emb_mask,
                example_weight=self.text_nas_weight)
            wr_loss, wr_emb_list = self.single_nce_loss(
                ridxs, self.review_emb, widxs, self.word_emb, self.word_bias,
                self.alias_words)
            loss_list.extend([ur_loss, pr_loss, wr_loss])
            l2_loss_list += ur_emb_list + pr_emb_list + wr_emb_list
        else:
            uw_loss, uw_emb_list = self.single_nce_loss(
                uidxs,
                self.user_emb,
                widxs,
                self.word_emb,
                self.word_bias,
                self.alias_words,
                example_mask=self.user_emb_mask,
                example_weight=self.text_nas_weight)
            pw_loss, pw_emb_list = self.single_nce_loss(
                pidxs,
                self.product_emb,
                widxs,
                self.word_emb,
                self.word_bias,
                self.alias_words,
                example_mask=self.product_emb_mask,
                example_weight=self.text_nas_weight)
            loss_list.extend([uw_loss, pw_loss])
            l2_loss_list += uw_emb_list + pw_emb_list
        loss = sum(loss_list)
        return loss, l2_loss_list

    def single_nce_loss(self,
                        example_idxs,
                        example_emb,
                        label_idxs,
                        label_emb,
                        label_bias,
                        label_alias,
                        example_mask=None,
                        label_mask=None,
                        example_weight=None,
                        label_weight=None):
        # negative sampling
        sampled_ids = label_alias.draw(self.negative_samples)

        example_vec = self.get_embeddings(
            example_emb, example_idxs, example_mask,
            example_weight)  # example_emb[example_idxs]
        true_w = self.get_embeddings(label_emb, label_idxs, label_mask,
                                     label_weight)  # label_emb[label_idxs]
        true_b = self.get_embeddings(label_bias, label_idxs, label_mask,
                                     label_weight)

        sampled_w = self.get_embeddings(label_emb, sampled_ids, label_mask,
                                        label_weight)  # label_emb[sampled_ids]
        sampled_b = self.get_embeddings(label_bias, sampled_ids, label_mask,
                                        label_weight)

        true_logit = torch.sum(example_vec.mul(true_w), dim=-1) + true_b
        sampled_logit = torch.matmul(example_vec, sampled_w.transpose(
            0, 1)) + sampled_b

        return self.nce_loss(true_logit,
                             sampled_logit), [example_vec, true_w, sampled_w]

    def nce_loss(self, true_logit, sampled_logit):
        true = F.binary_cross_entropy_with_logits(true_logit,
                                                  torch.ones_like(true_logit))
        false = F.binary_cross_entropy_with_logits(
            sampled_logit, torch.zeros_like(sampled_logit))
        nce_loss = true + false
        return nce_loss

    @staticmethod
    def l2_loss(t):
        return torch.sum(t**2) / 2

    def get_embeddings(self, embeddings, idxes, masks=None, weight=None):
        ret = emb = embeddings[idxes]
        if masks is not None:
            if weight is not None:
                ret = emb * masks * weight
            else:
                ret = emb * masks
        return ret

    def get_combined_features(self, example_idxs, embedding_list,
                              embedding_masks):
        op_n = self.combine_func
        op_w = self._op_w
        output = None
        for name, weight in zip(op_n, op_w):
            if weight > 0:
                if name == 'concat':
                    output = get_concat_embedding(example_idxs, embedding_list,
                                                  embedding_masks) * weight
                else:
                    tmp_list = [None for i in range(len(embedding_list))]
                    for i in range(len(embedding_list)):
                        tmp_list[i] = embedding_list[i][example_idxs]
                        if embedding_masks is not None:
                            tmp_list[i] = tmp_list[i] * embedding_masks[i]
                    if name == 'add':
                        output = sum(tmp_list) * weight
                    elif name == 'max':
                        tmp_list = torch.cat(
                            [i.unsqueeze(1) for i in tmp_list], dim=1)
                        output = torch.max(tmp_list, dim=1)[0] * weight
                    elif name == "mean":
                        sum_mask = sum(embedding_masks)
                        sum_mask[sum_mask == 0] = 1
                        output = sum(tmp_list) / sum_mask * weight
                    elif name == "min":
                        tmp_list = torch.cat(
                            [i.unsqueeze(1) for i in tmp_list], dim=1)
                        output = torch.min(tmp_list, dim=1)[0] * weight
                    else:
                        raise NotImplementedError
        return output.squeeze()
