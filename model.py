from monai.networks.nets import DenseNet121, EfficientNetBN, resnet10, resnet18, resnet34, resnet50, ViT
from comparison import *

def get_opt(args, blocks, channels):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = blocks # number of basic blocks in the backbone
            self.channels = channels # number of channels of deep features
            self.n_classes = args.num_classes # Dimension of out_channels
            self.emb_dims = 1024 # Dimension of embeddings
            self.relative_pos = args.relative_pos # Use relative pos
            self.use_backbone_only = args.use_backbone_only # Use relative pos
            self.which_backbone_stage = args.which_backbone_stage # Use relative pos
            
            
    return OptInit()

def get_model(args):
    
    print(args.model)
    
    if args.model == 'RGNN':
        model = get_RGNN(args)  
        
    elif args.model == 'ViG':
        model = get_ViG(args)   
        
    elif 'PRGNN' in args.model:
        model = get_PRGNN(args)   

    elif args.model == 'densenet':
        model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=args.num_classes)
        
    elif args.model == 'efficientnet':
        model = EfficientNetBN(model_name='efficientnet-b0', spatial_dims=3, in_channels=1, num_classes=args.num_classes)

    elif args.model == 'resnet10':
        model = resnet10(n_input_channels=1, num_classes=args.num_classes)
    elif args.model == 'resnet18':
        model = resnet18(n_input_channels=1, num_classes=args.num_classes)
    elif args.model == 'resnet34':
        model = resnet34(n_input_channels=1, num_classes=args.num_classes)
    elif args.model == 'resnet50':
        model = resnet50(n_input_channels=1, num_classes=args.num_classes)
        
    elif args.model == 'convnext_small':
        model = create_mednext_v1(num_input_channels=1, num_classes=args.num_classes, model_id='S', kernel_size=7)
    elif args.model == 'convnext_base':
        model = create_mednext_v1(num_input_channels=1, num_classes=args.num_classes, model_id='B', kernel_size=7)
        
    elif args.model == 'M3T':
        model = M3T(
            n_classes = args.num_classes
        )
        
    elif args.model == 'LRPM3T':
        model = LRP_M3T(
            n_classes = args.num_classes
        )
        
    elif args.model == 'PViG_ti':
        opt = get_opt(args, blocks=[2,2,6,2], channels=[48,96,240,384])
        model = PyramidViG(opt)
        
    elif args.model == 'PViG_s':
        opt = get_opt(args, blocks=[2,2,6,2], channels=[72,144,288,432])
        model = PyramidViG(opt)
        
    elif args.model == 'PViG_m':
        opt = get_opt(args, blocks=[2,2,6,2], channels=[96,192,384,768])
        model = PyramidViG(opt)
        
    elif args.model == 'PViG_b':
        opt = get_opt(args, blocks=[2,2,6,2], channels=[132, 252, 516, 1020])
        model = PyramidViG(opt)
        
    elif args.model == 'AAGN':
        model = AAGN(args)
    # elif args.model == 'M3T':
    #     model = M3T(
    #         n_classes = args.num_classes
    #     )
        
    elif args.model == 'ViT':
        model = ViT(in_channels=1, num_classes=args.num_classes, patch_size=4,
                    img_size=(96, 96, 96), proj_type='conv', pos_embed_type='sincos', classification=True)
        
    else:
        raise Exception('[ERROR] model name not recognized')

    return model
        
def get_ViG(args):
    from vig import DeepGCN
    
    class OptInit:
        def __init__(self, num_classes=4, drop_path_rate=0.0, drop_rate=0.0, num_knn=9):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 6 # number of basic blocks in the backbone
            self.n_filters = 96 # number of channels of deep features
            self.n_classes = args.num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.use_attention_pool = args.use_attention_pool # Attention pool
            self.drop_path = drop_path_rate
            self.one_hot_mask = 'template/AAL_add_midbrain.nii'
            self.device = f'cuda:{args.gpu}'

    opt = OptInit()
    model = DeepGCN(opt)
    return model



def get_RGNN(args):
    from rgnn import DeepGCN
    
    class OptInit:
        def __init__(self, num_classes=4, drop_path_rate=0.0, drop_rate=0.0):
            self.k = args.k # neighbor num (default:8)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = args.act # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = args.n_blocks # number of basic blocks in the backbone
            self.n_filters = args.n_filters # number of channels of deep features
            self.n_classes = args.num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.pool = args.pool # Attention pool
            self.use_stochastic = False # stochastic for gcn, True or False            
            self.one_hot_mask = 'template/AAL_reduced_mask.nii'
            self.drop_path = args.drop_path_rate
            self.device = f'cuda:{args.gpu}'

    opt = OptInit(args)
    model = DeepGCN(opt)
    return model


def get_PRGNN(args):
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
            # self.one_hot_mask = 'template/AAL_reduced_mask.nii'
            self.one_hot_mask = 'template/AAL_add_midbrain.nii'
            self.batch_size = args.batch_size
            self.relative_pos = args.relative_pos
            self.stage = args.stage
            self.use_backbone_only = args.use_backbone_only
            self.which_backbone_stage = args.which_backbone_stage
            
            # 새로 추가해야할것들
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