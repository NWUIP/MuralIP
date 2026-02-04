import os
from PIL import Image
import argparse
from guided_diffusion.metrics_my1 import EdgeAccuracy
from torchvision.utils import save_image
import numpy as np
import torch as th
import functools
import torch.distributed as dist
from guided_diffusion.my_util import stitch_images, create_dir, stitch_images1
from guided_diffusion.dataset_my import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from MAE.util import misc
from guided_diffusion.gaussian_diffusion import GaussianDiffusion

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16', random_mask=False, finetune=False):
    model = misc.get_mae_image_encoder_model(type='mae_vit_b')
    checkpoint = th.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print("sam.checkpoint:", checkpoint.keys())
    print("msg:", msg)
    return model

def main(model_path, save_dir):
    th.cuda.device_count()

    args = create_argparser().parse_args()
    args1 = create_argparser().parse_args()
    args1.in_channels = 6
    args1.out_channels = 3
    args1.model_path = model_path

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    model1, diffusion1 = create_model_and_diffusion(
        **args_to_dict(args1, model_and_diffusion_defaults().keys())
    )
    state_dict1 = th.load(args1.model_path, map_location=th.device('cpu'))
    model1.load_state_dict(state_dict1, strict=True)

    if len(model1.load_state_dict(state_dict1)) == 0:
        print("All parameters have been loaded successfully.")
    else:
        print("There are some parameters that were not loaded.")

    model1.to(dist_util.dev())
    mae_chkpt_dir = ''
    mae_model = prepare_model(mae_chkpt_dir, random_mask=False, finetune=False).to(dist_util.dev())

    if args.use_fp16:
        model1.convert_to_fp16()
    model1.eval()
    import os
    from transformers import (
        CLIPTokenizer, CLIPTextModel, CLIPVisionModel, CLIPImageProcessor
    )

    local_dir = ""

    os.environ["HF_HUB_OFFLINE"] = "1"

    tokenizer = CLIPTokenizer.from_pretrained(local_dir, local_files_only=True)
    tokenizer.model_max_length = 77
    caption_clip_model = CLIPTextModel.from_pretrained(local_dir, local_files_only=True).to("cuda:1").eval()
    clip_img_proc = CLIPImageProcessor.from_pretrained(local_dir, local_files_only=True)
    image_clip_model = CLIPVisionModel.from_pretrained(local_dir, local_files_only=True).to("cuda:1").eval()
    test_data = load_data(
        data_dir=args.test_datadir,
        mask_dir=args.test_maskdir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=False,
        deterministic=True,
        mask_train=False,
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print("complete.")

    all_images = []

    while len(all_images) * args.batch_size < args.num_samples:
        image, mask, mask_image, edge, mask_edge, gray_maskimage, gray_image, mae_yuantu, mae_mask,caption,caption_tokenize,caption_mask, _ = next(test_data)
        image = image.to(dist_util.dev())
        mask = mask.to(dist_util.dev())
        mask_image = mask_image.to(dist_util.dev())
        edge = edge.to(dist_util.dev())
        mask_edge = mask_edge.to(dist_util.dev())
        gray_maskimage = gray_maskimage.to(dist_util.dev())
        gray_image = gray_image.to(dist_util.dev())
        mae_yuantu = mae_yuantu.to(dist_util.dev())
        mae_mask = mae_mask.to(dist_util.dev())
        caption_tokenize = caption_tokenize.to(dist_util.dev())
        caption_mask = caption_mask.to(dist_util.dev())
        model_kwargs = {}

        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        logger.log("image sampling...")

        jiazao_time1 = th.tensor([249])
        tensor1 = jiazao_time1.long().to(dist_util.dev())

        compute_losses1 = functools.partial(
            diffusion1.training_losses11,
            model1,
            mae_model,

            image,
            mask_image,

            mae_yuantu,
            mae_mask,

            caption_tokenize,
            caption_mask,
            caption_clip_model,
            image_clip_model,
            clip_img_proc,


            tensor1
        )

        sample3= compute_losses1()


        sample33 = sample3
        sample33 = ((sample33 + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample33 = sample33.permute(0, 2, 3, 1)
        sample33 = sample33.contiguous()
        image1 = Image.fromarray(sample33.squeeze().cpu().numpy())
        idx = len(all_images)
        file_name = f'{idx:05}.png'
        file_path = os.path.join(save_dir, file_name)
        image1.save(file_path)

        all_images.append("1")

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=150,
        batch_size=1,
        use_ddim=False,
        test_datadir="",
        test_maskdir="",
        image_size=256,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    steps = []
    for step in steps:
        model_path = f""
        save_dir = f""
        main(model_path, save_dir)