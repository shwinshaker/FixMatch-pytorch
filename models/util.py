

def create_model(args):
    if args.arch == 'wideresnet':
        from .wideresnet import build_wideresnet
        model = build_wideresnet(depth=args.model_depth,
                                 widen_factor=args.model_width,
                                 dropout=0,
                                 num_classes=args.num_classes)
    elif args.arch == 'resnext':
        from .resnext import build_resnext
        model = build_resnext(cardinality=args.model_cardinality,
                              depth=args.model_depth,
                              width=args.model_width,
                              num_classes=args.num_classes)
    return model