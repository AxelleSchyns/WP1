import numpy as np
import torch
import torch.nn as nn


class SBSM(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(SBSM, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, window_size, stride, savepath='masks.npy'):
        """
        Generates sliding window type binary masks used in augment() to 
        mask an image. The Images are resized to 224x224 to 
        enable re-use of masks Generating the sliding window style masks.
        :param int window_size: the block window size 
        (with value 0, other areas with value 1)
        :param int stride: the sliding step
        :param tuple image_size: the mask size which should be the 
        same to the image size
        :return: the sliding window style masks
        :rtype: numpy.ndarray
        """

        rows = np.arange(0 + stride - window_size, self.input_size[0], stride)
        cols = np.arange(0 + stride - window_size, self.input_size[1], stride)

        mask_num = len(rows) * len(cols)
        masks = np.ones(
            (mask_num, self.input_size[0], self.input_size[1]), dtype=np.uint8)
        i = 0
        for r in rows:
            for c in cols:
                if r < 0:
                    r1 = 0
                else:
                    r1 = r
                if r + window_size > self.input_size[0]:
                    r2 = self.input_size[0]
                else:
                    r2 = r + window_size
                if c < 0:
                    c1 = 0
                else:
                    c1 = c
                if c + window_size > self.input_size[1]:
                    c2 = self.input_size[1]
                else:
                    c2 = c + window_size
                masks[i, r1:r2, c1:c2] = 0
                i += 1
        masks = masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, masks)
        self.register_buffer('masks', torch.from_numpy(masks).cuda())
        self.N = self.masks.shape[0]
        self.window_size = window_size
        self.stride = stride

    def load_masks(self, filepath):
        masks = np.load(filepath)
        self.register_buffer('masks', torch.from_numpy(masks).cuda())
        self.N = self.masks.shape[0]

    def weighted_avg(self, K):
        count = self.N - self.masks.sum(dim=(0, 1))
        sal = K.sum(dim=-1).permute(2, 0, 1) / count

        return sal

    def forward(self, x_q, x):
        with torch.no_grad():
            x = x.cuda()
            # Get embedding of query and retrieval image
            x_q = self.model.get_vector(x_q)
            x_r = self.model.get_vector(x)
            o_dist = torch.cdist(x_q, x_r)
            # Apply array of masks to the image
            stack = torch.mul(self.masks, x)

            x = []
            for i in range(0, self.N, self.gpu_batch):
                x.append(self.model(stack[i:min(i + self.gpu_batch, self.N)]))
            x = torch.cat(x)
            m_dist = torch.cdist(x_q, x)

            # Compute saliency
            K = (1 - self.masks).permute(2, 3, 1, 0) * \
                (m_dist - o_dist).clamp(min=0)
            sal = self.weighted_avg(K)

        return sal


class SBSMBatch(SBSM):
    def forward(self, x_q, x=None):
        if x is None:
            x = x_q
            self_sim = True
        else:
            self_sim = False
        B, C, H, W = x.size()

        with torch.no_grad():
            # Get embedding of query and retrieval images
            x_q = self.model(x_q)
            if not self_sim:
                x_r = self.model(x)
                o_dist = torch.cdist(x_q, x_r)
                o_dist = o_dist.view(-1, 1)

            # Apply array of masks to the image
            stack = torch.mul(self.masks.view(self.N, 1, H, W),
                              x.view(B * C, H, W))
            stack = stack.view(B * self.N, C, H, W)

            x = []
            for i in range(0, self.N*B, self.gpu_batch):
                x.append(self.model(
                    stack[i:min(i + self.gpu_batch, self.N*B)]))
            x = torch.cat(x)
            if self_sim:
                m_dist = torch.norm(x_q.unsqueeze(
                    1) - x.view(-1, B, x_q.shape[1]).permute(1, 0, 2), dim=2)
            else:
                m_dist = torch.cdist(x_q, x)

            # Compute saliency
            if self_sim:
                K = (1 - self.masks).permute(2, 3, 1, 0) * m_dist
            else:
                m_dist = m_dist.view(-1, self.N, B).permute(0,
                                                            2, 1).reshape(-1, self.N)
                K = (1 - self.masks).permute(2, 3, 1, 0) * \
                    (m_dist - o_dist).clamp(min=0)
            sal = self.weighted_avg(K)

        return sal

