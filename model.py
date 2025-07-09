def get_model(args):
    from prgnn import DeepGCN
    
    class OptInit:
        def __init__(self, num_classes=4, drop_path_rate=0.0, drop_rate=0.0, num_knn=9):
            self.k = args.k # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_classes = args.num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.pool = 'avgpool' # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.one_hot_mask = args.roi_mask
            self.batch_size = args.batch_size
            self.relative_pos = False
            
            if args.model == 'PRGNN_ti':
                self.blocks = [2, 2, 6, 2]
                self.in_channels = [48, 96, 144, 336]
                self.channels = [48, 48, 96, 240]
                
            if args.model == 'PRGNN_s':
                self.blocks = [2, 2, 6, 2]
                self.in_channels = [72, 144, 216, 432]
                self.channels = [72, 72, 144, 288]
                
            if args.model == 'PRGNN_m':
                self.blocks = [2, 2, 16, 2]
                self.in_channels = [96, 192, 288, 576]
                self.channels = [96, 96, 192, 384]
                
            if args.model == 'PRGNN_b':
                self.blocks = [2, 2, 18, 2]
                self.in_channels = [144, 288, 432, 864]
                self.channels = [144, 144, 288, 576]
                
            if args.dataparallel:
                self.device = f'cuda'
            else:
                self.device = f'cuda:{args.gpu}'

    opt = OptInit(args)
    model = DeepGCN(opt)
    return model