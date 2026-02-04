
"""
Train a diffusion model on images.
"""
import os

import argparse
from guided_diffusion.dataset_my import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from torch.utils.tensorboard import SummaryWriter
import torch
from guided_diffusion.discriminator import Discriminator
from MAE.util import misc
import os

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16', random_mask=False, finetune=False):

    model = misc.get_mae_image_encoder_model(type='mae_vit_b')

    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print("sam.checkpoint:",checkpoint.keys())

    print("msg:",msg)
    return model


def main():
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.device_count()
    args = create_argparser().parse_args()
    args1 = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    print(args1.in_channels)
    args1.in_channels=6
    args1.out_channels=3

    print(args_to_dict(args, model_and_diffusion_defaults().keys()) )
    print(args_to_dict(args1, model_and_diffusion_defaults().keys()) )

    mae_chkpt_dir = ''

    mae_model = prepare_model(mae_chkpt_dir, random_mask=False, finetune=False).to("cpu")
    for param in mae_model.parameters():
        param.requires_grad = False
    model1, diffusion1 = create_model_and_diffusion(
        **args_to_dict(args1, model_and_diffusion_defaults().keys())
    )


    discriminator1 = Discriminator(in_channels=3, use_sigmoid=True)


    mae_model.to(dist_util.dev())
    model1.to(dist_util.dev())
    discriminator1.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion1)

    logger.log("creating data loader...")

    print("batchsize:",args.batch_size)

    import os
    from transformers import CLIPTokenizer, CLIPTextModel

    local_dir = ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["HF_HUB_OFFLINE"] = "1"

    tokenizer = CLIPTokenizer.from_pretrained(local_dir, local_files_only=True)
    tokenizer.model_max_length = 77
    text_model = CLIPTextModel.from_pretrained(local_dir, local_files_only=True).to(device).eval()
    data = load_data(
        data_dir=args.data_dir,
        mask_dir=args.mask_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=False,
        mask_train=True

    )
    print("$$$")

    val_data = load_data(
        data_dir=args.valdata_dir,
        mask_dir=args.mask_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        mask_train=True,
    )


    logger.log("training...")
    print("---------------------------------")
    TrainLoop(
        model1=model1,
        diffusion1=diffusion1,
        discriminator1=discriminator1,
        mae_model=mae_model,
        data=data,
        val_data=val_data,

        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint11=args.resume_checkpoint11,
        resume_checkpoint21=args.resume_checkpoint21,
        resume_checkpoint31=args.resume_checkpoint31,
        resume_checkpoint41=args.resume_checkpoint41,
        resume_checkpoint51=args.resume_checkpoint51,


        use_fp16=False,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,

    ).run_loop()



def create_argparser():
    defaults = dict(

        data_dir="",
        valdata_dir="",
        mask_dir="",
        schedule_sampler="uniform",

        lr=1e-6,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=-1,
        ema_rate="0.9999",

        log_interval=5000,
        save_interval=5000,
        resume_checkpoint11="",
        resume_checkpoint21="",
        resume_checkpoint31="",
        resume_checkpoint41="",
        resume_checkpoint51="",

        use_fp16=False,
        fp16_scale_growth=1e-3,

        resume_step="",
        model_path="",
        image_input_channels=6,
        image_output_channels=3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
