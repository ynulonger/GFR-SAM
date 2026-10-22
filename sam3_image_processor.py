# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
from typing import Dict, List

import numpy as np
import PIL
import torch

from sam3.model import box_ops

from sam3.model.data_misc import FindStage, interpolate
from torchvision.transforms import v2
import torchvision
from .geometry_encoders import Prompt
from .geometry_encoders import concat_padded_sequences
from .act_ckpt_utils import activation_ckpt_wrapper


class Sam3Processor:
    """ """

    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        self.model = model
        self.resolution = resolution
        self.device = device
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @torch.inference_mode()
    def set_image(self, image, state=None):
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        image = v2.functional.to_image(image).to(self.device)
        image = self.transform(image).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_image_batch(self, images: List[np.ndarray], state=None):
        """Sets the image batch on which we want to do predictions."""
        if state is None:
            state = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(
            images[0], PIL.Image.Image
        ), "Images must be a list of PIL images"

        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]

        images = [
            self.transform(v2.functional.to_image(image).to(self.device))
            for image in images
        ]
        images = torch.stack(images, dim=0)
        state["backbone_out"] = self.model.backbone.forward_image(images)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict):
        """Sets the text prompt and run the inference"""

        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        # add unsapmeled mask
        state["unsampled_masks"] = out_masks # [N, 288, 288]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state


    @torch.inference_mode()
    def extract_exemplar_visual_feature(self, box: List, label: bool, state: Dict):
        """Extracts the visual feature for an exemplar box.
        The image needs to be set.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before extract_exemplar_visual_feature")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)
        # Encode prompts immediately
        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            prompt, prompt_mask, backbone_out = self.model._encode_prompt(
                backbone_out=state["backbone_out"],
                find_input=self.find_stage,
                geometric_prompt=state["geometric_prompt"]
            )
        
        # Store encoded prompts directly in state
        state["prompt"] = prompt
        state["prompt_mask"] = prompt_mask
        state["backbone_out"] = backbone_out
        
        # Remove geometric_prompt as we only rely on prompt and prompt_mask for inference
        # if "geometric_prompt" in state:
        #     del state["geometric_prompt"]

        return state


    @torch.inference_mode()
    def _forward_grounding_by_vision(self, state: Dict):
        # Get image features 用来更新prompt
        feat_tuple = self.model._get_img_feats(state["backbone_out"], self.find_stage.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple
        seq_first_img_feats = img_feats[-1]  # [H*W, B, C]
        seq_first_img_pos_embeds = (
            img_pos_embeds[-1]
            if img_pos_embeds is not None
            else torch.zeros_like(seq_first_img_feats)
        )

        # print("enable _geometry_encoder forward to update prompt")
        final_embed = state["boxes_embed"]
        encoder = self.model.geometry_encoder
        for lay in encoder.encode:
            final_embed = activation_ckpt_wrapper(lay)(
                tgt=final_embed,
                memory=seq_first_img_feats,
                tgt_key_padding_mask=state["geo_masks"],
                pos=seq_first_img_pos_embeds,
                act_ckpt_enable=encoder.training and encoder.use_act_ckpt,
            )
        final_embed = encoder.encode_norm(final_embed)
        # print(f"final_embed shape: {final_embed.shape}")
        # print(final_embed.shape)


        # 组装
        prompt = torch.cat([state['txt_feats'], final_embed, state['visual_prompt_embed']], dim=0)
        prompt_mask = torch.cat([state['txt_masks'], state['geo_masks'], state['visual_prompt_mask']], dim=1)
        state["prompt"] = prompt
        state["prompt_mask"] = prompt_mask

        # Run the encoder
        with torch.profiler.record_function("SAM3Image._run_encoder"):
            backbone_out, encoder_out, _ = self.model._run_encoder(
                backbone_out=state["backbone_out"], 
                find_input=self.find_stage, 
                prompt=state["prompt"], 
                prompt_mask=state["prompt_mask"]
            )
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }

        

        # Run the decoder
        with torch.profiler.record_function("SAM3Image._run_decoder"):
            out, hs = self.model._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=state["prompt"],
                prompt_mask=state["prompt_mask"],
                encoder_out=encoder_out,
            )

        # Run segmentation heads
        with torch.profiler.record_function("SAM3Image._run_segmentation_heads"):
            self.model._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=self.find_stage.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=state["prompt"],
                prompt_mask=state["prompt_mask"],
                hs=hs,
            )

        outputs = out

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        # 保留采样前的mask
        state["unsampled_masks"] = out_masks
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]        
        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state

    def my_reset_all_prompts_by(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "prompt", "prompt_mask", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]


    @torch.inference_mode()
    def extract_exemplar_visual_feature_v2(self, box: List, label: bool, state: Dict, text_prompt='visual'):
        """Extracts the visual feature for an exemplar box.
        The image needs to be set.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        """

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                [text_prompt], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)


        txt_ids = self.find_stage.text_ids
        txt_feats = state["backbone_out"]["language_features"][:, txt_ids]
        txt_masks = state["backbone_out"]["language_mask"][txt_ids]


        # encoder stage
        gen_encoder = self.model.geometry_encoder

        feat_tuple = self.model._get_img_feats(state["backbone_out"], self.find_stage.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        # print(f"img_feats shape is {img_feats[0].shape}")
    
        # Encode prompts immediately
        # Encode geometry
        geo_feats, geo_masks, un_tf_final_embed = self.gen_encoder_forward(
            encoder=gen_encoder,
            geo_prompt=state["geometric_prompt"],
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
            img_masks=state['unsampled_masks'] if 'unsampled_masks' in state else None
        )
        
        # Append zero visual prompt embeddings
        visual_prompt_embed = torch.zeros(
            (0, *geo_feats.shape[1:]), device=geo_feats.device
        )
        visual_prompt_mask = torch.zeros(
            (*geo_masks.shape[:-1], 0),
            device=geo_masks.device,
            dtype=geo_masks.dtype,
        )
        
        # encode text prompt
        if state.get("only_visual_prompt", False):
            prompt = torch.cat([geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([geo_masks, visual_prompt_mask], dim=1)
        else:
            prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)

        # prompt = torch.cat([geo_feats], dim=0)
        # prompt_mask = torch.cat([geo_masks], dim=1)

        # print(f"text feats shape is {txt_feats.shape}")
        # print(f"text masks shape is {txt_masks.shape}")
        # print(f"geo_feats shape is {geo_feats.shape}")
        # print(f"geo_masks shape is {geo_masks.shape}")
        # print(f"prompt shape is {prompt.shape}")
        # print(f"prompt_mask shape is {prompt_mask.shape}")

        # print(f"txt_masks is {txt_masks}")
        # print(f"geo_masks is {geo_masks}")
        # print(f"prompt_mask is {prompt_mask}")


        # Store encoded prompts directly in state
        state["prompt"] = prompt
        state["prompt_mask"] = prompt_mask
        state["backbone_out"] = backbone_out

        state["txt_feats"] = txt_feats
        state["txt_masks"] = txt_masks
        state["geo_feats"] = geo_feats
        state["geo_masks"] = geo_masks
        state["visual_prompt_embed"] = visual_prompt_embed
        state["visual_prompt_mask"] = visual_prompt_mask
        state["boxes_embed"] = un_tf_final_embed

        return state

    def gen_encoder_forward(self, encoder, geo_prompt: Prompt, img_feats, img_sizes, img_pos_embeds=None,img_masks=None):
        points = geo_prompt.point_embeddings
        points_mask = geo_prompt.point_mask
        points_labels = geo_prompt.point_labels
        boxes = geo_prompt.box_embeddings
        boxes_mask = geo_prompt.box_mask
        boxes_labels = geo_prompt.box_labels
        masks = geo_prompt.mask_embeddings
        masks_mask = geo_prompt.mask_mask
        masks_labels = geo_prompt.mask_labels
        seq_first_img_feats = img_feats[-1]  # [H*W, B, C]
        seq_first_img_pos_embeds = (
            img_pos_embeds[-1]
            if img_pos_embeds is not None
            else torch.zeros_like(seq_first_img_feats)
        )

        if encoder.points_pool_project or encoder.boxes_pool_project:
            assert len(img_feats) == len(img_sizes)
            cur_img_feat = img_feats[-1]
            cur_img_feat = encoder.img_pre_norm(cur_img_feat)
            H, W = img_sizes[-1]
            assert cur_img_feat.shape[0] == H * W
            N, C = cur_img_feat.shape[-2:]
            # Put back in NxCxHxW
            cur_img_feat = cur_img_feat.permute(1, 2, 0)
            cur_img_feat = cur_img_feat.view(N, C, H, W)
            img_feats = cur_img_feat

        final_embeds, final_mask = encoder._encode_points(
            points=points,
            points_mask=points_mask,
            points_labels=points_labels,
            img_feats=img_feats,
        )
        

        # 去除几何信息
        if not encoder.encode_boxes_as_points:
            boxes_embeds, boxes_mask = self.gen_encoder_encode_box(
                encoder=encoder,
                boxes=boxes,
                boxes_mask=boxes_mask,
                boxes_labels=boxes_labels,
                img_feats=img_feats,
                img_masks=img_masks,
            )

            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, boxes_embeds, boxes_mask
            )
            # print(f"boxes embeds shape is {boxes_embeds.shape}")
            # print(f"boxes mask shape is {boxes_mask.shape}")
            # print(f"boxes_mask is {boxes_mask}")

        if masks is not None and encoder.mask_encoder is not None:
            masks_embed, masks_mask = encoder._encode_masks(
                masks=masks,
                attn_mask=masks_mask,
                mask_labels=masks_labels,
                img_feats=img_feats,
            )
            if points.size(0) == boxes.size(0) == 0:
                return masks_embed, masks_mask
        bs = final_embeds.shape[1]
        assert final_mask.shape[0] == bs
        if encoder.cls_embed is not None:
            cls = encoder.cls_embed.weight.view(1, 1, encoder.d_model).repeat(1, bs, 1)
            cls_mask = torch.zeros(
                bs, 1, dtype=final_mask.dtype, device=final_mask.device
            )
            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, cls, cls_mask
            )

        if encoder.final_proj is not None:
            final_embeds = encoder.norm(encoder.final_proj(final_embeds))
        untf_final_embeds = final_embeds
        # print(f"untf_final_embeds shape: {untf_final_embeds.shape}")
        if encoder.encode is not None:
            for lay in encoder.encode:
                final_embeds = activation_ckpt_wrapper(lay)(
                    tgt=final_embeds,
                    memory=seq_first_img_feats,
                    tgt_key_padding_mask=final_mask,
                    pos=seq_first_img_pos_embeds,
                    act_ckpt_enable=encoder.training and encoder.use_act_ckpt,
                )
            final_embeds = encoder.encode_norm(final_embeds)
        # Finally, concat mask embeddings if any
        if masks is not None and encoder.mask_encoder is not None:
            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, masks_embed, masks_mask
            )
        
        return final_embeds, final_mask, untf_final_embeds # 论文里的
        # return final_embeds, final_mask, untf_final_embeds, boxes_embeds # MAP 消融



    def gen_encoder_encode_box(self, encoder, boxes, boxes_mask, boxes_labels, img_feats, img_masks=None):
        boxes_embed = None
        n_boxes, bs = boxes.shape[:2]

        # 直接投影
        # if encoder.boxes_direct_project is not None:
        #     print("*"*30)
        #     print("Using direct box projection")
        #     print("*"*30)
        #     proj = encoder.boxes_direct_project(boxes)
        #     assert boxes_embed is None
        #     boxes_embed = proj

        if encoder.boxes_pool_project is not None:
            H, W = img_feats.shape[-2:]
            # print("*" * 30)
            # print("Using box ROI pooling")
            # print(f"img_feats shape is {img_feats.shape}")
            # print(f"img_masks shape is {img_masks.shape if img_masks is not None else None}")
            # print("*" * 30)
            # boxes are [Num_boxes, bs, 4], normalized in [0, 1]
            # We need to denormalize, and convert to [x, y, x, y]
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
            scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype)
            scale = scale.pin_memory().to(device=boxes_xyxy.device, non_blocking=True)
            scale = scale.view(1, 1, 4)
            boxes_xyxy = boxes_xyxy * scale

            # mask * feature
            if img_masks is not None:
                img_masks.to(device=img_feats.device)
                img_masks = interpolate(
                    img_masks.unsqueeze(1),
                    (H, W),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid() > 0.5  # B x 1 x H x W
                merged_mask_bool = torch.any(img_masks, dim=0, keepdim=True)
                # print(f"merged_mask_bool shape is {merged_mask_bool.shape}")
                img_feats = img_feats * merged_mask_bool
                
            sampled = torchvision.ops.roi_align(
                img_feats, boxes_xyxy.float().transpose(0, 1).unbind(0), encoder.roi_size
            )
            # print(f"sampled shape: {sampled.shape}")
            assert list(sampled.shape) == [
                bs * n_boxes,
                encoder.d_model,
                encoder.roi_size,
                encoder.roi_size,
            ]
            proj = encoder.boxes_pool_project(sampled)
            proj = proj.view(bs, n_boxes, encoder.d_model).transpose(0, 1)
            # print(f"proj shape: {proj.shape}")
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        # if encoder.boxes_pos_enc_project is not None:
        #     print("*"*30)
        #     print("Using box positional encoding")
        #     print("*"*30)
        #     cx, cy, w, h = boxes.unbind(-1)
        #     enc = encoder.pos_enc.encode_boxes(
        #         cx.flatten(), cy.flatten(), w.flatten(), h.flatten()
        #     )
        #     enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])

        #     proj = encoder.boxes_pos_enc_project(enc)
        #     if boxes_embed is None:
        #         boxes_embed = proj
        #     else:
        #         boxes_embed = boxes_embed + proj

        type_embed = encoder.label_embed(boxes_labels.long())
        assert  torch.equal(boxes_embed, proj)
        # return boxes_embed + type_embed, boxes_mask # 测试加type_embed的效果
        return boxes_embed, boxes_mask #论文数据没加type_embed
    