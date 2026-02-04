




import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def adjust_discriminator_lr(dis_optimizer, generator_lr, dis_lr_factor=0.1):
    """调整判别器优化器的学习率，使其为生成器优化器学习率的固定倍数"""
    new_lr = generator_lr * dis_lr_factor
    for param_group in dis_optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def adjust_learning_rate_finetune(optimizer, epoch, args, start_from=0):
    """Decay the learning rate with half-cycle cosine after warmup"""
    epoch = epoch - start_from
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - start_from - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr