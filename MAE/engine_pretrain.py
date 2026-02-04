








import math
import os
import sys
from typing import Iterable
import torch.nn.functional as F
import cv2
import numpy as np
import torch
import torch.nn as nn
import MAE.util.lr_sched as lr_sched
import MAE.util.misc as misc
from MAE.loss import AdversarialLoss,PerceptualLoss,StyleLoss


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    dis_model=None,
                    dis_optimizer=None
                    ):
    model.train(True)
    dis_model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    print("accum_iter:",accum_iter)
    optimizer.zero_grad()
    dis_optimizer.zero_grad()
    first = True

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    adversarial_loss = AdversarialLoss(type="nsgan")
    perceptual_loss = PerceptualLoss()
    style_loss = StyleLoss()
    l1_loss = nn.L1Loss()

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
        dis_optimizer.zero_grad()













        for k in samples:
            print(k)
            if type(samples[k]) is torch.Tensor:
                samples[k] = samples[k].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            terms={}

            imgs, pred, mask = model(samples['img'],samples['img_1024'], samples['mask'],samples['mask_1024'], mask_ratio=args.mask_ratio)
            print("shuchu....")
            print("imgs.shape:",imgs.shape)
            print("pred.shape:",pred.shape)
            print("mask.shape:",mask.shape)
            print("samples['mask'].shape:",samples['mask'].shape)








            dis_input_real = imgs
            dis_input_fake = pred.detach()
            dis_real, _ = dis_model(dis_input_real)

            dis_fake, _ = dis_model(dis_input_fake)

            dis_real_loss = adversarial_loss(dis_real, True, True)
            dis_fake_loss = adversarial_loss(dis_fake, False, True)
            dis_loss = (dis_real_loss + dis_fake_loss) / 2

            terms["dis_loss"] = dis_loss
            print("dis_loss:",dis_loss)
            dis_loss /= accum_iter
            dis_loss.backward()
            dis_optimizer.step()



            gen_l2_loss = mean_flat((imgs - pred) ** 2).mean()
            gen_l2_loss = gen_l2_loss * 20
            gen_loss = gen_l2_loss

            terms["gen_l2_loss"] = gen_l2_loss
            print("gen_l2_loss:", gen_l2_loss)

            gen_l1_loss = l1_loss(pred * samples['mask'], imgs * samples['mask'])
            gen_l1_loss = gen_l1_loss * 20
            gen_loss = gen_loss + gen_l1_loss

            terms["gen_l1_loss"] = gen_l1_loss
            print("gen_l1_loss:", gen_l1_loss)

            gen_input_fake = pred
            gen_fake, _ = dis_model(gen_input_fake)
            gen_gan_loss = adversarial_loss(gen_fake, True, False) * 100
            terms["gen_gan_loss"] = gen_gan_loss
            gen_loss = gen_loss + gen_gan_loss
            print("gen_gan_loss:", gen_gan_loss)
            gen_content_loss = perceptual_loss(pred, imgs)
            gen_content_loss = gen_content_loss * 0.35
            terms["gen_perceptual_loss"] = gen_content_loss
            gen_loss = gen_loss + gen_content_loss
            print("gen_content_loss:", gen_content_loss)






            print("gen_loss:", gen_loss)

        loss_value = gen_loss.item()
        dis_loss=terms["dis_loss"].item()
        gen_l2_loss=terms["gen_l2_loss"].item()
        gen_l1_loss=terms["gen_l1_loss"].item()
        gen_gan_loss=terms["gen_gan_loss"].item()
        gen_perceptual_loss=terms["gen_perceptual_loss"].item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        gen_loss /= accum_iter


        loss_scaler(gen_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)




        print("(data_iter_step + 1) % accum_iter:",(data_iter_step + 1) % accum_iter)




        torch.cuda.synchronize(device=1)

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        dis_loss_reduce = misc.all_reduce_mean(dis_loss)
        gen_l2_loss_reduce = misc.all_reduce_mean(gen_l2_loss)
        gen_l1_loss_reduce = misc.all_reduce_mean(gen_l1_loss)
        gen_gan_loss_reduce = misc.all_reduce_mean(gen_gan_loss)
        gen_perceptual_loss_reduce = misc.all_reduce_mean(gen_perceptual_loss)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('gen_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('dis_loss', dis_loss_reduce, epoch_1000x)
            log_writer.add_scalar('gen_l2_loss', gen_l2_loss_reduce, epoch_1000x)
            log_writer.add_scalar('gen_l1_loss', gen_l1_loss_reduce, epoch_1000x)
            log_writer.add_scalar('gen_gan_loss', gen_gan_loss_reduce, epoch_1000x)
            log_writer.add_scalar('gen_perceptual_loss', gen_perceptual_loss_reduce, epoch_1000x)

            log_writer.add_scalar('lr', lr, epoch_1000x)

        if data_iter_step % 100 == 0:
            print("----first and is_process----")
            first = False
            print("hasattr(model, 'module'):",hasattr(model, 'module'))
            model_without_ddp = model.module if hasattr(model, 'module') else model
            os.makedirs(args.output_dir + '/samples', exist_ok=True)

            y = pred[:4]
            y = torch.einsum('nchw->nhwc', y).detach()


            mask = mask[:4].detach()

            mask = mask.unsqueeze(-1).repeat(1, 1, model_without_ddp.patch_size ** 2 * 3)

            mask = model_without_ddp.unpatchify(mask)
            print("mask.shape:",mask.shape)
            resized_mask = mask
            print("resized_mask.shape:", resized_mask.shape)




            mask = torch.einsum('nchw->nhwc', resized_mask).detach()


            x = torch.einsum('nchw->nhwc', samples['img'][:4])
            mask_data=torch.einsum('nchw->nhwc', samples["mask"][:4])
            im_masked = x * (1 - mask_data)
            im_masked = im_masked.cpu()
            print("im_masked.shape:",im_masked.shape)
            im_masked = torch.cat(tuple(im_masked), dim=0)
            print("im_masked.shape:", im_masked.shape)
            im_paste = x * (1 - mask_data) + y * mask_data
            im_paste = im_paste.cpu()
            im_paste = torch.cat(tuple(im_paste), dim=0)
            print("im_paste.shape:",im_paste.shape)

            im_masked1 = x * (1 - mask)
            im_masked1 = im_masked1.cpu()
            print("im_masked1.shape:",im_masked1.shape)
            im_masked1 = torch.cat(tuple(im_masked1), dim=0)
            print("im_masked1.shape:", im_masked1.shape)

            print("mask.shape:",mask.shape)

            im_paste1 = x * (1 - mask) + y * mask
            im_paste1 = im_paste1.cpu()
            im_paste1 = torch.cat(tuple(im_paste1), dim=0)
            print("im_paste1.shape:",im_paste1.shape)

            x = x.cpu()
            y = y.cpu()
            x = torch.cat(tuple(x), dim=0)
            y = torch.cat(tuple(y), dim=0)
            print("x.shape:",x.shape)
            print("y.shape:",y.shape)
            images = torch.cat([x.float(), im_masked.float(), y.float(), im_paste.float(),im_masked1.float(),im_paste1.float()], dim=1)
            images = torch.clip((images * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255, 0, 255).int()
            images = images.numpy().astype(np.uint8)

            path = os.path.join(args.output_dir, 'samples')
            name = os.path.join(path, str(epoch).zfill(5)+'_'+str(data_iter_step).zfill(10) + ".jpg")
            print('\nsaving sample ' + name)

            print("images.shape:",images.shape)
            cv2.imwrite(name, images[:, :, ::-1])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_finetune(model: torch.nn.Module,
                             data_loader: Iterable, optimizer, optimizer_new,
                             device: torch.device, epoch: int, loss_scaler,
                             log_writer=None, args=None, start_epoch=0):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('alpha', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    optimizer_new.zero_grad()
    first = True

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate_finetune(optimizer, data_iter_step / len(data_loader) + epoch, args, start_from=start_epoch)
            lr_sched.adjust_learning_rate_finetune(optimizer_new, data_iter_step / len(data_loader) + epoch, args, start_from=start_epoch)
        for k in samples:
            if type(samples[k]) is torch.Tensor:
                samples[k] = samples[k].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, pred, mask, irr_mask, partial_mask = model(samples['img'], samples['mask'], mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, optimizer_new, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            optimizer_new.zero_grad()

        torch.cuda.synchronize(device=1)

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(alpha=model.module.alpha.item())

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('alpha', model.module.alpha.item(), epoch_1000x)

        if first and misc.is_main_process():
            first = False
            os.makedirs(args.output_dir + '/samples', exist_ok=True)
            model_without_ddp = model.module if hasattr(model, 'module') else model
            y = model_without_ddp.unpatchify(pred[:4])
            y = torch.einsum('nchw->nhwc', y).detach()
            mask = mask[:4].detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, model_without_ddp.patch_embed.patch_size[0] ** 2 * 3)
            mask = model_without_ddp.unpatchify(mask)
            mask = torch.einsum('nchw->nhwc', mask).detach()

            partial_mask = partial_mask[:4].detach()
            partial_mask = partial_mask.repeat(1, 1, model_without_ddp.patch_embed.patch_size[0] ** 2 * 3)
            partial_mask = model_without_ddp.unpatchify(partial_mask)
            partial_mask = torch.einsum('nchw->nhwc', partial_mask).detach()

            irr_mask = irr_mask[:8].detach()
            irr_mask = torch.einsum('nchw->nhwc', irr_mask).detach()

            x = torch.einsum('nchw->nhwc', samples['img'][:8])
            im_masked = x * (1 - mask)
            im_masked = im_masked.cpu()
            im_masked = torch.cat(tuple(im_masked), dim=0)

            im_masked2 = x * (1 - irr_mask)
            im_masked2 = im_masked2.cpu()
            im_masked2 = torch.cat(tuple(im_masked2), dim=0)

            im_masked3 = x * (1 - partial_mask)
            im_masked3 = im_masked3.cpu()
            im_masked3 = torch.cat(tuple(im_masked3), dim=0)
            im_paste = x * (1 - mask) + y * mask
            im_paste = im_paste.cpu()
            im_paste = torch.cat(tuple(im_paste), dim=0)
            x = x.cpu()
            y = y.cpu()
            x = torch.cat(tuple(x), dim=0)
            y = torch.cat(tuple(y), dim=0)

            images = torch.cat([x.float(), im_masked.float(), im_masked2.float(), im_masked3.float(), y.float(), im_paste.float()], dim=1)
            images = torch.clip((images * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255, 0, 255).int()
            images = images.numpy().astype(np.uint8)

            path = os.path.join(args.output_dir, 'samples')
            name = os.path.join(path, str(epoch).zfill(10) + ".jpg")
            print('\nsaving sample ' + name)
            cv2.imwrite(name, images[:, :, ::-1])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
