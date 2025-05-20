import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module import *
from flash_attn import flash_attn_func


class SpatialAttn(nn.Module):
    def __init__(
        self, q_dim, kv_dim, hid_dim, heads, device, attn_drop=0.
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.heads = heads
        self.head_dim = hid_dim // heads
        self.scale = self.head_dim ** -0.5

        self.spa_q_conv = nn.Conv2d(q_dim, self.hid_dim, kernel_size=1, bias=False, device=device)
        self.spa_k_conv = nn.Conv2d(kv_dim, self.hid_dim, kernel_size=1, bias=False, device=device)
        self.spa_v_conv = nn.Conv2d(kv_dim, self.hid_dim, kernel_size=1, bias=False, device=device)
        self.out_drop = nn.Dropout(attn_drop)
        
    # cls_map.shape: (B, T, N, num_reg)
    def forward(self, x, cls_map=None):
        B, T, N, D = x.shape

        spatial_q = self.spa_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        spatial_k = self.spa_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)     # B, T, N, head_dim * num_heads
        spatial_v = self.spa_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)     # B, T, N, head_dim * num_heads

        spatial_q = spatial_q.view(B, T, N, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)
        spatial_k = spatial_k.view(B, T, N, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)
        spatial_v = spatial_v.view(B, T, N, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)
        spa_attn = (spatial_q @ spatial_k.transpose(-2, -1)) * self.scale   # (B, T, num_heads, N, N)
        # spa_attn represents the hihg-level correlations
        if cls_map is not None:
            spa_attn = cls_map.unsqueeze(2).repeat(1, 1, self.heads, 1, 1).transpose(-1, -2) @ spa_attn
        spa_attn = spa_attn.softmax(dim=-1)
        spa_attn = self.out_drop(spa_attn)   # (B, T, H, num_reg, N)

        spa_x = (spa_attn @ spatial_v).transpose(2, 3).reshape(B, T, cls_map.shape[-1], self.heads * self.head_dim)
        return spa_x


class TemporalAttn(nn.Module):
    def __init__(
        self, q_dim, kv_dim, hid_dim, heads, device, agg=False, num_reg=None, seg_num=None, attn_drop=0.
    ):
        super().__init__()
        assert hid_dim % heads == 0
        assert not agg or seg_num is not None, "If it is aggregative, seg_num must be given."

        self.hid_dim = hid_dim
        self.heads = heads
        self.head_dim = hid_dim // heads
        self.scale = self.head_dim ** -0.5
        self.agg = agg
        self.seg_num = seg_num

        if self.agg:
            self.temporal_q = nn.Parameter(torch.randn(size=(num_reg, self.seg_num, self.hid_dim), device=device))
        else:
            self.tmp_q_conv = nn.Conv2d(q_dim, self.hid_dim, kernel_size=1, bias=False, device=device)

        self.tmp_k_conv = nn.Conv2d(kv_dim, self.hid_dim, kernel_size=1, bias=False, device=device)
        self.tmp_v_conv = nn.Conv2d(kv_dim, self.hid_dim, kernel_size=1, bias=False, device=device)
        self.out_drop = nn.Dropout(attn_drop)

    # tmp_info.shape: (B, N, T, P)
    def forward(self, x, tmp_info=None):
        B, T, N, D = x.shape

        if self.agg:
            temporal_q = self.temporal_q.unsqueeze(0).repeat(B, 1, 1, 1)  # B, N, P, head_dim * num_heads
            attn_T = self.seg_num
        else:
            temporal_q = self.tmp_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1) # B, N, T, head_dim * num_heads
            attn_T = T
        temporal_k = self.tmp_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)     # B, N, T, head_dim * num_heads
        temporal_v = self.tmp_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)     # B, N, T, head_dim * num_heads

        temporal_q = temporal_q.view(B, N, attn_T, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)
        temporal_k = temporal_k.view(B, N, T, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)
        temporal_v = temporal_v.view(B, N, T, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)
        tmp_attn = (temporal_q @ temporal_k.transpose(-2, -1)) * self.scale   # B, N, H, P, T

        if tmp_info is not None:
            tmp_attn = tmp_info.unsqueeze(2).repeat(1, 1, self.heads, 1, 1) @ tmp_attn
        tmp_attn = tmp_attn.softmax(dim=-1)
        tmp_attn = self.out_drop(tmp_attn)

        temporal_x = tmp_attn @ temporal_v    # (B, N, num_heads, T, head_dim)
        temporal_x = temporal_x.transpose(2, 3).reshape(B, N, T, self.heads * self.head_dim).transpose(1, 2)
        return temporal_x


class SpatialDiffAttn(nn.Module):
    def __init__(
        self, layer_idx, q_dim, kv_dim, hid_dim, heads, device, attn_drop=0.
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.heads = heads
        self.hid_dim = hid_dim
        self.head_dim = hid_dim // heads // 2
        self.scale = self.head_dim ** -0.5
        self.device = device

        self.spa_q_conv = nn.Conv2d(q_dim, self.hid_dim, kernel_size=1, bias=False, device=device)
        self.spa_k_conv = nn.Conv2d(kv_dim, self.hid_dim, kernel_size=1, bias=False, device=device)
        self.spa_v_conv = nn.Conv2d(kv_dim, self.hid_dim, kernel_size=1, bias=False, device=device)

        self.lambda_init = lambda_init(layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32, device=device).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32, device=device).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32, device=device).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32, device=device).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim).to(self.device)
        self.out_drop = nn.Dropout(attn_drop)


    def forward(self, x):
        B, T, N, D = x.shape

        spatial_q = self.spa_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # B, T, N, 2 * head_dim * num_heads
        spatial_k = self.spa_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # B, T, N, 2 * head_dim * num_heads
        spatial_v = self.spa_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # B, T, N, 2 * head_dim * num_heads

        spatial_q = spatial_q.contiguous().view(B * T, N, self.heads, 2 * self.head_dim)
        spatial_k = spatial_k.contiguous().view(B * T, N, self.heads, 2 * self.head_dim)
        spatial_v = spatial_v.contiguous().view(B * T, N, self.heads, 2 * self.head_dim)
        sq1, sq2 = spatial_q.chunk(2, dim=-1)
        sk1, sk2 = spatial_k.chunk(2, dim=-1)
        sv1, sv2 = spatial_v.chunk(2, dim=-1)
        
        attn11 = flash_attn_func(sq1.half(), sk1.half(), sv1.half())
        attn12 = flash_attn_func(sq1.half(), sk1.half(), sv2.half())
        attn1 = torch.cat([attn11, attn12], dim=-1).float()

        attn21 = flash_attn_func(sq2.half(), sk2.half(), sv1.half())
        attn22 = flash_attn_func(sq2.half(), sk2.half(), sv2.half())
        attn2 = torch.cat([attn21, attn22], dim=-1).float()
        
        lambda1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(spatial_q)
        lambda2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(spatial_q)
        lambda_full = lambda1 - lambda2 + self.lambda_init
        spa_attn = attn1 - lambda_full * attn2
        spa_attn = self.out_drop(spa_attn)

        spa_attn = self.subln(spa_attn)
        spa_attn = spa_attn * (1 - self.lambda_init)
        s_x = spa_attn.reshape(B, T, N, 2 * self.head_dim * self.heads)
        return s_x



class STAttention(nn.Module):

    def __init__(
        self, dim, s_heads, sa_heads, t_heads, ta_heads, dtw_map_param, 
        num_reg, seg_cls_num, layer_idx, attn_drop, agg_drop, device
    ):
        super().__init__()
        avg_dim = dim // (s_heads + sa_heads + t_heads + ta_heads)
        s_dim = s_heads * avg_dim
        sa_dim = sa_heads * avg_dim
        t_dim = t_heads * avg_dim
        ta_dim = ta_heads * avg_dim
        
        self.dtw_map_param = dtw_map_param
        self.spa_attn = SpatialDiffAttn(
            layer_idx=layer_idx, q_dim=dim, kv_dim=dim, hid_dim=s_dim, 
            heads=s_heads, device=device, attn_drop=attn_drop
        )
        self.dtw_agg_attn = nn.ModuleList([
            SpatialAttn(
                q_dim=dim, kv_dim=dim, hid_dim=sa_dim, 
                heads=sa_heads, device=device, attn_drop=agg_drop
            )
            for _ in range(len(self.dtw_map_param))
        ])
        self.spa_agg_linear = nn.Linear(sa_dim, sa_dim, device=device)

        self.tmp_attn = TemporalAttn(
            q_dim=dim, kv_dim=dim, hid_dim=t_dim, 
            heads=t_heads, device=device, attn_drop=attn_drop
        )
        self.tmp_agg_attn = TemporalAttn(
            q_dim=dim, kv_dim=dim, hid_dim=ta_dim, heads=ta_heads, device=device, agg=True, 
            num_reg=num_reg, seg_num=seg_cls_num, attn_drop=agg_drop
        )

        self.out_proj = nn.Linear(s_dim + sa_dim + t_dim + ta_dim, dim, bias=False, device=device)
        

    def forward(self, x, dtw_agg_x, tmp_inf):
        B, T, _, _ = x.shape
        s_x = self.spa_attn(x)
        
        sg_x = 0.
        for i, map_param in enumerate(self.dtw_map_param):
            cls_map = map_param.unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)   # (B, T, M, N)
            h = self.dtw_agg_attn[i](dtw_agg_x[i], cls_map)    
            sg_x = sg_x + h
        sg_x = self.spa_agg_linear(sg_x)
        
        t_x = self.tmp_attn(x)
        tg_x = self.tmp_agg_attn(x, tmp_inf)   # (B, P, N, d)

        st_x = torch.cat([s_x, sg_x, t_x, tg_x], dim=-1)
        st_x = self.out_proj(st_x)
        return st_x



class STEncoder(nn.Module):

    def __init__(
        self, dim, ext_dim, 
        s_heads, sa_heads, t_heads, ta_heads,
        dtw_map, num_reg, seg_cls_num,
        layer_idx, mlp_ratio, drop_rate, attn_drop, agg_drop, device
    ):
        super().__init__()
        self.device = device
        # param list: eg.(16, 263), (8, 263), used to aggregate/seperate regions.
        self.dtw_map_param = self.get_map_param(dtw_map)
        self.tmp_map_linear = nn.Linear(ext_dim, seg_cls_num, device=self.device)

        self.norm1 = nn.LayerNorm(dim).to(device=device)
        self.st_attn = STAttention(dim=dim, s_heads=s_heads, sa_heads=sa_heads, t_heads=t_heads, ta_heads=ta_heads,
                                   dtw_map_param=self.dtw_map_param,
                                   num_reg=num_reg, seg_cls_num=seg_cls_num, 
                                   layer_idx=layer_idx, attn_drop=attn_drop, agg_drop=agg_drop, device=device)
        
        self.norm2 = nn.LayerNorm(dim).to(device=device)
        self.mlp = Mlp(dim, hidden_features=int(mlp_ratio * dim)).to(self.device)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        

    # aggx list: eg.[[B,T,16,d], [B,T,8,d]]
    def forward(self, x, dtw_agg_x, add_inf):
        tmp_map = self.tmp_map_linear(add_inf.transpose(1, 2))   # (B, N, T, P)

        sh = self.st_attn(x, dtw_agg_x, tmp_map)
        h = self.drop_path(sh) + x
        h = self.norm1(h)
        y = self.drop_path(self.mlp(h)) + h
        y = self.norm2(y)
        return y
    
    def get_map_param(self, cls_maps):
        map_params = nn.ParameterList()
        for map in cls_maps:
            forward_map = torch.randn(map.size(), device=self.device) * map
            map_params.append(nn.Parameter(forward_map))
        return map_params
       
    
        


class ADFormer(nn.Module):

    def __init__(self, args, dataset_feature, logger):
        super().__init__()

        self.args = args
        self.device = args.device
        self.dataset_feature = dataset_feature
        self.logger = logger
        self.ext_dim = self.dataset_feature.get("ext_dim", 8)
        self.scaler = self.dataset_feature.get("scaler")
        self.data_dim = args.output_dim
        
        # spatial embedding
        self.spatial_emb = nn.Parameter(torch.randn((args.num_reg, args.SE_dim), device=self.device))
        self.data_embedding = DataEmbedding(
            self.data_dim,  args.embed_dim, args.SE_dim, device=args.device
        )
        self.cls_data_embedding = nn.ModuleList()
        self.spa_cls_emb = nn.ParameterList()
        cls_reg_nums = eval(args.cluster_reg_nums)
        for reg_num in cls_reg_nums:
            self.spa_cls_emb.append(nn.Parameter(torch.randn((reg_num, args.SE_dim), device=self.device)))
            self.cls_data_embedding.append(DataEmbedding(
                self.data_dim, args.embed_dim, args.SE_dim, device=args.device
            ))

        # 0-1 tensor list, is used to aggregate X.
        self.dtw_map = self.dataset_feature.get("dtw_map")
        
        drops = [x.item() for x in torch.linspace(0, args.drop_path, args.depth)]
        self.enc_blocks = nn.ModuleList([
            STEncoder(dim=args.embed_dim, ext_dim=self.ext_dim,
                      s_heads=args.s_heads, sa_heads=args.sa_heads, t_heads=args.t_heads, ta_heads=args.ta_heads,
                      dtw_map=self.dtw_map, 
                      num_reg=args.num_reg, seg_cls_num=args.cluster_seg_num,
                      layer_idx=idx+1, mlp_ratio=args.mlp_ratio, drop_rate=drops[idx], 
                      attn_drop=args.attn_drop, agg_drop=args.agg_drop, device=args.device)
            for idx in range(args.depth)
        ])

        self.skip_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=args.embed_dim, out_channels=args.skip_dim, kernel_size=1, device=self.device),
            )
            for _ in range(args.depth)
        ])
        self.end_conv1 = nn.Conv2d(
            in_channels=args.window, out_channels=args.horizon, kernel_size=1, bias=True, device=args.device
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=args.skip_dim, out_channels=args.output_dim, kernel_size=1, bias=True, device=args.device
        )
        
        
    
    def forward(self, x):
        B, T, _, _ = x.shape
        add_inf = x[..., self.data_dim:]   # (B, T, N, 8)

        
        dtw_agg_x = []
        for i, map in enumerate(self.dtw_map):
            h = map.unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1) @ x[..., :self.args.output_dim]
            inf = add_inf[:, :, 0, :].unsqueeze(-2).repeat(1, 1, h.shape[2], 1)
            h = torch.cat([h, inf], dim=-1)
            dtw_agg_x.append(self.cls_data_embedding[i](h, self.spa_cls_emb[i]))

        x = self.data_embedding(x, self.spatial_emb)   # (B, T, N, d)
        skip = 0.
        for i, block in enumerate(self.enc_blocks):
            x =  block(x, dtw_agg_x, add_inf)
            skip = skip + self.skip_convs[i](x.permute(0, 3, 2, 1))
            
        out = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        out = self.end_conv2(F.relu(out.permute(0, 3, 2, 1)))
        out = out.permute(0, 3, 2, 1)

        return out
    
    
