import torch.nn as nn
import torch


def soft_vale(r_, lam_):
    R = torch.sign(r_) * torch.clamp(torch.abs(r_) - lam_, 0)
    return R

class MSrSefAtt(nn.Module):
    def __init__(self, win_size, bands, num_heads):
        super().__init__()
        self.win_size = win_size
        self.bands = bands
        self.num_heads =num_heads
        head_bands = int(bands/num_heads)
        self.head_bands = head_bands

        # 1
        self.embedding_in = nn.Conv2d(bands, head_bands * num_heads,3,1, padding=1, bias=False)
        self.embedding_out = nn.Conv2d(head_bands * num_heads, bands,3,1, padding=1, bias=False)

        # 2
        self.W1_heads = nn.ModuleList([nn.Conv2d(win_size * win_size, win_size * win_size, 1, 1,bias=False) for _ in range(num_heads)])
        self.W2_heads = nn.ModuleList([nn.Conv2d(head_bands, win_size * win_size, 1, 1, bias=False) for _ in range(num_heads)])
        self.soft_thr1 = nn.Parameter(torch.ones(num_heads)*0.1 )
        self.soft_thr2 = nn.Parameter(torch.ones(num_heads)*0.1)
        self.relu = nn.ReLU()
        self.gn = nn.GroupNorm(1,bands,affine=False)

        # 3
        self.re_data = nn.ModuleList([nn.Conv2d(win_size*win_size, head_bands, 1, 1, bias=False) for _ in range(num_heads)])

    def forward(self, Y):
        Y_e = self.embedding_in(Y)
        Y = Y_e.view(-1,self.num_heads,self.head_bands, self.win_size, self.win_size)
        outputs = []
        for i in range(self.num_heads):
            # iteration-1
            R1 = self.W2_heads[i](Y[:,i])
            X1 = soft_vale(R1, self.soft_thr1[i])
            X1 = self.relu(X1)
            # iteration-2
            R2 = self.W1_heads[i](X1) + R1
            X2 = soft_vale(R2, self.soft_thr2[i])
            X2 = self.relu(X2)

            # reconstitution
            output = torch.bmm(Y[:,i].flatten(2), X2.flatten(2))
            output = output.view(Y[:,i].shape)
            outputs.append(output)

        multihead_output = torch.cat(outputs, dim=1)
        multihead_output = self.embedding_out(multihead_output)
        return multihead_output

class CSPD(nn.Module):
    def __init__(self, class_num, win_size, bands, drop_ratio=0.3):
        super().__init__()
        # f1
        self.f1 = nn.Conv2d(bands, int(bands/2), 3, stride=1, padding=1, bias=False)

        # f2
        self.f2 = nn.Conv2d(int(bands/2), int(bands/2), 5, 1, padding=2, bias=False)
        self.gn2 = nn.GroupNorm(1, int(bands/2), affine=False)
        # f3
        self.f3 = nn.Conv2d(int(bands/2), bands*class_num, 3, padding=1, stride=1, bias=False)

        self.class_num = class_num
        self.bands = bands
        self.win_size = win_size

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop_ratio)

    def forward(self,Y):
        Y1 = self.relu(self.gn2(self.f2(self.f1(Y))))
        Y1 = self.drop(Y1)
        Y1 = self.f3(Y1)

        Y1 = Y1.view(-1, self.class_num, self.bands, self.win_size, self.win_size)
        class_subs = Y1
        Y1 = torch.sum(Y1, dim=1)

        # F-loss
        res = Y - Y1
        res = torch.mean(torch.pow(res, 2))
        return Y1, res, class_subs

class MyBlock(nn.Module):
    def __init__(self,
                 win_size,
                 bands,
                 class_num,
                 num_heads,
                 drop_ratio = 0.3):


        super().__init__()
        self.gn = nn.GroupNorm(1,bands, affine=False)
        self.MSrSefAtt = MSrSefAtt(win_size, bands, num_heads)
        self.cspd = CSPD(class_num, win_size, bands, drop_ratio)
        self.drop = nn.Dropout(drop_ratio)

    def forward(self, Y):
        # part-1
        # Y1 = self.gn(Y)
        Y1 = self.MSrSefAtt(Y)
        Y1 = self.drop(Y1)

        # part-2
        Y2 = Y1+Y
        Y2 = self.gn(Y2)
        Y2, res_loss, class_token = self.cspd(Y2)
        Y2 = self.drop(Y2)

        # part-3
        Y3 = Y1+Y2
        return Y3, res_loss, class_token

class MyNetwork(nn.Module):
    def __init__(self,
                 win_size,
                 bands,
                 embedding_bands,
                 num_heads,
                 block_nums,
                 class_nums=16,
                 drop_rate=0.3):
        super().__init__()
        self.block_nums = block_nums
        # 01
        win_size1 = 9
        win_size2 = int((win_size + 1) / 2)
        if win_size1 > win_size2:
            win_size = win_size2
        else:
            win_size = win_size1

        self.embedding_layer = nn.Sequential(nn.Conv2d(bands, embedding_bands, 5,1, 2),
                                             nn.BatchNorm2d(embedding_bands),
                                             nn.AdaptiveAvgPool2d((win_size, win_size))
                                            )

        # 02
        self.drop = nn.Dropout(p=drop_rate)
        self.encoder = nn.ModuleList([MyBlock(win_size,embedding_bands,class_nums,num_heads,drop_rate) for _ in range(block_nums)])
        # 03
        self.decoder = nn.Sequential(
            nn.Conv3d(class_nums, class_nums,(embedding_bands,win_size,win_size)),
        )

    def forward(self,Y):
        # 01
        Y = self.embedding_layer(Y)

        # 02
        for i in range(self.block_nums):
            Y, loss_i, class_subs= self.encoder[i](Y)
            if i == 0:
                res_loss = loss_i
            else:
                res_loss = res_loss + loss_i
        res_loss = res_loss / self.block_nums

        # 03
        Y_res = torch.pow(class_subs, 2).flatten(2).mean(2)
        return Y_res, res_loss



