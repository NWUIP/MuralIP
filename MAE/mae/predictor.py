




import numpy as np
import torch

from .modeling import Sam

from typing import Optional, Tuple

from .utils.transforms import ResizeLongestSide
class SamPredictor:

    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model


        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)

        self.reset_image()
    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        如何计算提供图像的image embedding赋值给了self.features
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."

        if image_format != self.model.image_format:
            image = image[..., ::-1]



        print("image.shape:",image.shape)
        input_image = self.transform.apply_image(image)
        print("input_image.shape:",input_image.shape)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        print("input_image_torch.shape:", input_image_torch.shape)


        print("input_image_torch.permute(2, 0, 1).contiguous().shape:", input_image_torch.permute(2, 0, 1).shape)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        print("input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :].shape:",input_image_torch.shape)

        print("image.shape[:2]:",image.shape[:2])
        self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        该函数计算提供的图像的嵌入，允许使用“predict”方法预测掩码
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size

        self.input_size = tuple(transformed_image.shape[-2:])


        input_image = self.model.preprocess(transformed_image)



        self.features = self.model.image_encoder(input_image)
        print("**first stage self.features.shape:",self.features.shape)

        self.is_image_set = True


    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,

        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels. 每个点坐标以(X,Y)像素格式 point提示词
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point. 1为前景  0为背景
            1 表示该点是前景点，即这个点位于图像中的主要对象或感兴趣的区域上；
            0 表示该点是背景点，即这个点位于图像的主要对象之外的区域。
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format. box的  左上 右下的坐标 （shape）应该是 (4,)
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.模型的低分辨率掩码输入，通常来自前一次预测迭代
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
            决定了是否需要模型返回多个掩码，这在输入提示不明确时可能更有用
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask. 返回的是掩码的原始逻辑值还是经过阈值处理的二进制掩码

        Returns:
            预测的掩码，格式为CxHxW，C是掩码的数量，H和W是原始图像的尺寸。
            一个数组，包含模型对每个掩码质量的评分。
            一个低分辨率的掩码逻辑值数组，可以用于下一次迭代的掩码输入。
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """

        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."





            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)

            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:



            box = self.transform.apply_boxes(box, self.original_size)

            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]

        if mask_input is not None:

            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]




        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )
        print("---predict----------")
        print("masks.shape:",masks.shape)
        print("iou_predictions.shape:",iou_predictions.shape)
        print("low_res_masks.shape:",low_res_masks.shape)

        masks_np = masks[0].detach().cpu().numpy()
        print("masks_np.shape:",masks_np.shape)
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        print("iou_predictions_np.shape:", iou_predictions_np.shape)
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        print("low_res_masks_np.shape:", low_res_masks_np.shape)
        return masks_np, iou_predictions_np, low_res_masks_np

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.
        输入提示是批量的torch张量，并且预期已经使用ResizeLongestSide转换到输入框架。

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None



        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,


            image_pe=self.model.prompt_encoder.get_dense_pe(),

            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,

            multimask_output=multimask_output,
        )



        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        print("zhenshi mask:",low_res_masks)
        if not return_logits:
            masks = masks > self.model.mask_threshold
            print("self.model.mask_threshold:",self.model.mask_threshold)
            print("yuzhi chuli....")
            print("*yuzhi masks:",masks)
            unique_values = torch.unique(masks)
            print(unique_values)
        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image.  设置一些参数为none"""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
