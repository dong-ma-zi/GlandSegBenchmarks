import argparse

def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='sam_adpt', help='net type')
    # parser.add_argument('-baseline', type=str, default='unet', help='baseline net type')
    # parser.add_argument('-mod', type=str, default='sam_adpt', help='mod type:seg, cls, val_ad')
    parser.add_argument('-exp_name', type=str, default='monusac-samAdpt-b-1024-16-256', help='net type')
    # parser.add_argument('-type', type=str, default='map', help='condition type:ave,rand,rand_map')
    parser.add_argument('-vis', type=int, default=None, help='visualization')
    parser.add_argument('-reverse', type=bool, default=False, help='adversary reverse')
    parser.add_argument('-pretrain', type=bool, default=False, help='adversary reverse')
    parser.add_argument('-val_freq',type=int,default=5, help='interval between each validation')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=1, help='use which gpu')

    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')

    parser.add_argument('-weights', type=str, default=0, help='the weights file you want to test')
    parser.add_argument('-distributed', default='none', type=str,help='multi GPU ids to use')
    parser.add_argument('-sam_ckpt', default="/home/data1/my/Project/segment-anything-main/sam_vit_b.pth",
                        help='sam checkpoint address')

    opt = parser.parse_args()

    return opt
