import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--exp_id', type=str, default=None, help='id of Experiment')
    parser.add_argument('--dataset_city', type=str, default='XIAN')
    parser.add_argument('--dataset_name', type=str, default='XIAN-Taxi')

    # data config
    parser.add_argument('--time_interval', type=int, default=1800, help='interval of dataset, unit: s')
    parser.add_argument('--num_reg', type=int, default=135, help='region num')
    parser.add_argument('--load_external', type=bool, default=True, help='whether to add external feature to raw data')
    parser.add_argument('--window', type=int, default=6, help='input steps')
    parser.add_argument('--horizon', type=int, default=1, help='output steps')
    parser.add_argument('--output_dim', type=int, default=2, help='output dimension, 2 (pick_up and drop_off)')
    parser.add_argument('--train_rate', type=float, default=0.7)
    parser.add_argument('--val_rate', type=float, default=0.1)
    parser.add_argument('--metric_mask', type=float, default=0.5)

    # model config
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of data embedding')
    parser.add_argument('--skip_dim', type=int, default=256, help='dimension of skip connection')
    parser.add_argument('--SE_dim', type=int, default=8, help='Laplacian eigenvectors dimension')

    
    parser.add_argument('--s_heads', type=int, default=3)
    parser.add_argument('--sa_heads', type=int, default=2)
    parser.add_argument('--t_heads', type=int, default=2)
    parser.add_argument('--ta_heads', type=int, default=1)
    parser.add_argument('--cluster_seg_num', type=int, default=2)
    parser.add_argument('--bal_cls', type=bool, default=True)
    parser.add_argument('--cls_tolerance', type=float, default=0.7)
    parser.add_argument('--cluster_reg_nums', type=str, default='[64,16]', help='list of agg regions')
    
    parser.add_argument('--swiglu_ratio', type=float, default=8/3, help='hidden dim ratio in diff-attn model')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='hidden dim ratio in non-diff-attn model')
    parser.add_argument('--attn_drop', type=float, default=0., help='attn drop rate')
    parser.add_argument('--agg_drop', type=float, default=0.1, help='agg attn drop rate')
    parser.add_argument('--drop_path', type=float, default=0.3, help='max drop path rate')
    parser.add_argument('--depth', type=int, default=6)


    
    # training config
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_epsilon', type=float, default=1e-8)
    parser.add_argument('--lr_beta1', type=float, default=0.9)
    parser.add_argument('--lr_beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0.05)

    parser.add_argument('--lr_decay', type=eval, default=True)
    parser.add_argument('--lr_scheduler', type=str, default='WarmupCosineAnnealingLR')
    parser.add_argument('--lr_warmup_init', type=float, default=1e-6)
    parser.add_argument('--lr_warmup_epoch', type=int, default=5)
    parser.add_argument('--lr_T_max', type=int, default=60)
    parser.add_argument('--lr_eta_min', type=float, default=1e-4)   
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)

    parser.add_argument('--clip_grad_norm', type=eval, default=True)
    parser.add_argument('--max_grad_norm', type=int, default=5)
    parser.add_argument('--use_early_stop', type=eval, default=True)
    parser.add_argument('--patience', type=int, default=50)
    
    args = parser.parse_args()
    return args