import argparse
import os
import random
import stat
import sys
from copy import deepcopy
from glob import glob
from pathlib import Path

import torch
from deep_translator import GoogleTranslator
from icetk import icetk as tokenizer
from PIL import Image
from SwissArmyTransformer import mpu
from SwissArmyTransformer.generation.sampling_strategies import BaseStrategy
from SwissArmyTransformer.resources import auto_create
from torchvision.utils import save_image
from tqdm import tqdm

os.environ["SAT_HOME"] = "modelzoo/"
cogvideo_submodule = os.path.abspath(os.path.dirname(__file__)) + "/../../../submodules/CogVideo/"
sys.path.append(cogvideo_submodule)

# HACK to override hardcoded cluster_label2.npy path :(
for file in [
    cogvideo_submodule + "/sr_pipeline/dsr_sampling.py",
    cogvideo_submodule + "/coglm_strategy.py",
]:
    with open(file, "r") as f:
        txt = f.read().replace("np.load('cluster_label2.npy')", f"np.load('{cogvideo_submodule}/cluster_label2.npy')")
    with open(file, "w") as f:
        f.write(txt)

from coglm_strategy import CoglmStrategy
from models.cogvideo_cache_model import CogVideoCacheModel
from sr_pipeline import DirectSuperResolution

tokenizer.add_special_tokens(["<start_of_image>", "<start_of_english>", "<start_of_chinese>"])

FL = FRAME_LEN = 400
FN = FRAME_NUM = 5


def get_masks_and_position_ids_stage1(seqlen, textlen):
    # Attention mask (lower triangular).
    attention_mask = torch.ones((1, textlen + FL, textlen + FL))
    attention_mask[:, :textlen, textlen:] = 0
    attention_mask[:, textlen:, textlen:].tril_()
    attention_mask.unsqueeze_(1)

    # Unaligned version
    position_ids = torch.zeros(seqlen, dtype=torch.long)
    torch.arange(textlen, out=position_ids[:textlen], dtype=torch.long)
    torch.arange(512, 512 + seqlen - textlen, out=position_ids[textlen:], dtype=torch.long)
    position_ids = position_ids.unsqueeze(0)

    return attention_mask, position_ids


def get_masks_and_position_ids_stage2(seqlen, textlen):
    # Attention mask (lower triangular).
    attention_mask = torch.ones((1, textlen + FL, textlen + FL))
    attention_mask[:, :textlen, textlen:] = 0
    attention_mask[:, textlen:, textlen:].tril_()
    attention_mask.unsqueeze_(1)

    # Unaligned version
    position_ids = torch.zeros(seqlen, dtype=torch.long)
    position_ids[:textlen] = torch.arange(textlen, dtype=torch.long)
    position_ids[textlen : textlen + FL] = torch.arange(512, 512 + FL, dtype=torch.long)
    position_ids[textlen + FL : textlen + FL * 2] = torch.arange(512 + FL * 2, 512 + FL * 3, dtype=torch.long)
    position_ids[textlen + FL * 2 : textlen + FL * 3] = torch.arange(512 + FL * 4, 512 + FL * 5, dtype=torch.long)
    position_ids[textlen + FL * 3 : textlen + FL * 4] = torch.arange(512 + FL * 1, 512 + FL * 2, dtype=torch.long)
    position_ids[textlen + FL * 4 : textlen + FL * 5] = torch.arange(512 + FL * 3, 512 + FL * 4, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0)

    return attention_mask, position_ids


def my_update_mems(hiddens, mems_buffers, mems_indexs, limited_spatial_channel_mem, text_len, frame_len):
    if hiddens is None:
        return None, mems_indexs
    mem_num = len(hiddens)
    ret_mem = []
    with torch.no_grad():
        for id in range(mem_num):
            if hiddens[id][0] is None:
                ret_mem.append(None)
            else:
                if (
                    id == 0
                    and limited_spatial_channel_mem
                    and mems_indexs[id] + hiddens[0][0].shape[1] >= text_len + frame_len
                ):
                    if mems_indexs[id] == 0:
                        for layer, hidden in enumerate(hiddens[id]):
                            mems_buffers[id][layer, :, :text_len] = hidden.expand(mems_buffers[id].shape[1], -1, -1)[
                                :, :text_len
                            ]
                    new_mem_len_part2 = (mems_indexs[id] + hiddens[0][0].shape[1] - text_len) % frame_len
                    if new_mem_len_part2 > 0:
                        for layer, hidden in enumerate(hiddens[id]):
                            mems_buffers[id][layer, :, text_len : text_len + new_mem_len_part2] = hidden.expand(
                                mems_buffers[id].shape[1], -1, -1
                            )[:, -new_mem_len_part2:]
                    mems_indexs[id] = text_len + new_mem_len_part2
                else:
                    for layer, hidden in enumerate(hiddens[id]):
                        mems_buffers[id][layer, :, mems_indexs[id] : mems_indexs[id] + hidden.shape[1]] = hidden.expand(
                            mems_buffers[id].shape[1], -1, -1
                        )
                    mems_indexs[id] += hidden.shape[1]
                ret_mem.append(mems_buffers[id][:, :, : mems_indexs[id]])
    return ret_mem, mems_indexs


def my_save_multiple_images(imgs, path, subdir, debug=True):
    # imgs: list of tensor images
    if debug:
        imgs = torch.cat(imgs, dim=0)
        save_image(imgs, path, normalize=True)
    else:
        single_frame_path = os.path.join(path, subdir)
        os.makedirs(single_frame_path, exist_ok=True)
        for i in range(len(imgs)):
            save_image(imgs[i], os.path.join(single_frame_path, f'{str(i).rjust(4,"0")}.jpg'), normalize=True)
            os.chmod(
                os.path.join(single_frame_path, f'{str(i).rjust(4,"0")}.jpg'),
                stat.S_IRWXO + stat.S_IRWXG + stat.S_IRWXU,
            )
        save_image(torch.cat(imgs, dim=0), os.path.join(single_frame_path, f"frame_concat.jpg"), normalize=True)
        os.chmod(os.path.join(single_frame_path, f"frame_concat.jpg"), stat.S_IRWXO + stat.S_IRWXG + stat.S_IRWXU)


def calc_next_tokens_frame_begin_id(text_len, frame_len, total_len):
    # The fisrt token's position id of the frame that the next token belongs to;
    if total_len < text_len:
        return None
    return (total_len - text_len) // frame_len * frame_len + text_len


mem_dict = {"len": None, "buffer": None, "guide_buffer": None}


def sample_token_sequence(
    model,
    seq,
    get_masks_and_position_ids,
    text_len,
    num_layers,
    hidden_size,
    keep_mem_buffers,
    strategy=BaseStrategy(),
    strategy2=BaseStrategy(),
    mems=None,
    log_text_attention_weights=0,  # default to 0: no artificial change
    mode_stage1=True,
    enforce_no_swin=False,
    guide_seq=None,
    guide_text_len=0,
    guidance_alpha=1,
    limited_spatial_channel_mem=False,  # 空间通道的存储限制在本帧内
    **kw_args,
):
    batch_size = 1
    print(
        f"{seq=}",
        f"{get_masks_and_position_ids=}",
        f"{text_len=}",
        f"{num_layers=}",
        f"{hidden_size=}",
        f"{keep_mem_buffers=}",
        f"{strategy=}",
        f"{strategy2=}",
        f"{mems=}",
        f"{log_text_attention_weights=}",
        f"{mode_stage1=}",
        f"{enforce_no_swin=}",
        f"{guide_seq=}",
        f"{guide_text_len=}",
        f"{guidance_alpha=}",
        f"{limited_spatial_channel_mem=}",
        f"{kw_args=}",
    )
    """
    seq: [2, 3, 5, ..., -1(to be generated), -1, ...]
    mems: [num_layers, batch_size, len_mems(index), mem_hidden_size]
        cache, should be first mems.shape[1] parts of context_tokens.
        mems are the first-level citizens here, but we don't assume what is memorized.
        input mems are used when multi-phase generation.
    """
    # building the initial tokens, attention_mask, and position_ids
    actual_context_length = 0

    while seq[-1][actual_context_length] >= 0:  # the last seq has least given tokens
        actual_context_length += 1  # [0, context_length-1] are given
    assert actual_context_length > 0
    current_frame_num = (actual_context_length - text_len) // FL
    assert current_frame_num >= 0
    context_length = text_len + current_frame_num * FL

    attention_mask, position_ids = get_masks_and_position_ids(len(seq[0]), text_len)
    attention_mask, position_ids = attention_mask.to(seq.device), position_ids.to(seq.device)
    tokens = seq[..., :context_length]
    input_tokens = tokens.clone()

    if guide_seq is not None:
        guide_index_delta = text_len - guide_text_len
        guide_attention_mask, guide_position_ids = get_masks_and_position_ids(len(guide_seq[0]), guide_text_len)
        guide_attention_mask, guide_position_ids = guide_attention_mask.to(seq.device), guide_position_ids.to(
            seq.device
        )
        guide_tokens = guide_seq[..., : context_length - guide_index_delta]
        guide_input_tokens = guide_tokens.clone()

    for fid in range(current_frame_num):
        input_tokens[:, text_len + 400 * fid] = tokenizer["<start_of_image>"]
        if guide_seq is not None:
            guide_input_tokens[:, guide_text_len + 400 * fid] = tokenizer["<start_of_image>"]

    attention_mask = attention_mask.type_as(next(model.parameters()))  # if fp16

    # initialize generation
    counter = context_length - 1  # Last fixed index is ``counter''
    index = 0  # Next forward starting index, also the length of cache.
    mems_buffers_on_GPU = False
    mems_indexs = [0, 0]
    mems_len = [(400 + 74) if limited_spatial_channel_mem else 5 * 400 + 74, 5 * 400 + 74]
    if keep_mem_buffers:
        mems_buffers = mem_dict["buffer"]
        for idx, mem_buffer in enumerate(mems_buffers):
            mems_buffers[idx] *= 0
    else:
        mems_buffers = [
            torch.zeros(num_layers, batch_size, mem_len, hidden_size * 2, dtype=next(model.parameters()).dtype)
            for mem_len in mems_len
        ]

    if guide_seq is not None:
        guide_attention_mask = guide_attention_mask.type_as(next(model.parameters()))  # if fp16
        if keep_mem_buffers:
            guide_mems_buffers = mem_dict["guide_buffer"]
            for idx, guide_mem_buffer in enumerate(guide_mems_buffers):
                guide_mems_buffers[idx] *= 0
        else:
            guide_mems_buffers = [
                torch.zeros(num_layers, batch_size, mem_len, hidden_size * 2, dtype=next(model.parameters()).dtype)
                for mem_len in mems_len
            ]
        guide_mems_indexs = [0, 0]
        guide_mems = None

    torch.cuda.empty_cache()
    # step-by-step generation
    with tqdm(total=int((seq == -1).sum()) + 1) as progress:
        while counter < len(seq[0]) - 1:
            # we have generated counter+1 tokens
            # Now, we want to generate seq[counter + 1],
            # token[:, index: counter+1] needs forwarding.
            if index == 0:
                group_size = 2 if (input_tokens.shape[0] == batch_size and not mode_stage1) else batch_size

                logits_all = None
                for batch_idx in range(0, input_tokens.shape[0], group_size):
                    logits, *output_per_layers = model(
                        input_tokens[batch_idx : batch_idx + group_size, index:],
                        position_ids[..., index : counter + 1],
                        attention_mask,  # TODO memlen
                        mems=mems,
                        text_len=text_len,
                        frame_len=FL,
                        counter=counter,
                        log_text_attention_weights=log_text_attention_weights,
                        enforce_no_swin=enforce_no_swin,
                        **kw_args,
                    )
                    logits_all = torch.cat((logits_all, logits), dim=0) if logits_all is not None else logits
                    mem_kv01 = [
                        [o["mem_kv"][0] for o in output_per_layers],
                        [o["mem_kv"][1] for o in output_per_layers],
                    ]
                    next_tokens_frame_begin_id = calc_next_tokens_frame_begin_id(text_len, FL, mem_kv01[0][0].shape[1])
                    for id, mem_kv in enumerate(mem_kv01):
                        for layer, mem_kv_perlayer in enumerate(mem_kv):
                            if limited_spatial_channel_mem and id == 0:
                                mems_buffers[id][
                                    layer, batch_idx : batch_idx + group_size, :text_len
                                ] = mem_kv_perlayer.expand(min(group_size, input_tokens.shape[0] - batch_idx), -1, -1)[
                                    :, :text_len
                                ]
                                mems_buffers[id][
                                    layer,
                                    batch_idx : batch_idx + group_size,
                                    text_len : text_len + mem_kv_perlayer.shape[1] - next_tokens_frame_begin_id,
                                ] = mem_kv_perlayer.expand(min(group_size, input_tokens.shape[0] - batch_idx), -1, -1)[
                                    :, next_tokens_frame_begin_id:
                                ]
                            else:
                                mems_buffers[id][
                                    layer, batch_idx : batch_idx + group_size, : mem_kv_perlayer.shape[1]
                                ] = mem_kv_perlayer.expand(min(group_size, input_tokens.shape[0] - batch_idx), -1, -1)
                    mems_indexs[0], mems_indexs[1] = mem_kv01[0][0].shape[1], mem_kv01[1][0].shape[1]
                    if limited_spatial_channel_mem:
                        mems_indexs[0] -= next_tokens_frame_begin_id - text_len

                mems = [mems_buffers[id][:, :, : mems_indexs[id]] for id in range(2)]
                logits = logits_all

                # guide
                if guide_seq is not None:
                    guide_logits_all = None
                    for batch_idx in range(0, guide_input_tokens.shape[0], group_size):
                        guide_logits, *guide_output_per_layers = model(
                            guide_input_tokens[batch_idx : batch_idx + group_size, max(index - guide_index_delta, 0) :],
                            guide_position_ids[
                                ..., max(index - guide_index_delta, 0) : counter + 1 - guide_index_delta
                            ],
                            guide_attention_mask,
                            mems=guide_mems,
                            text_len=guide_text_len,
                            frame_len=FL,
                            counter=counter - guide_index_delta,
                            log_text_attention_weights=log_text_attention_weights,
                            enforce_no_swin=enforce_no_swin,
                            **kw_args,
                        )
                        guide_logits_all = (
                            torch.cat((guide_logits_all, guide_logits), dim=0)
                            if guide_logits_all is not None
                            else guide_logits
                        )
                        guide_mem_kv01 = [
                            [o["mem_kv"][0] for o in guide_output_per_layers],
                            [o["mem_kv"][1] for o in guide_output_per_layers],
                        ]
                        for id, guide_mem_kv in enumerate(guide_mem_kv01):
                            for layer, guide_mem_kv_perlayer in enumerate(guide_mem_kv):
                                if limited_spatial_channel_mem and id == 0:
                                    guide_mems_buffers[id][
                                        layer, batch_idx : batch_idx + group_size, :guide_text_len
                                    ] = guide_mem_kv_perlayer.expand(
                                        min(group_size, input_tokens.shape[0] - batch_idx), -1, -1
                                    )[
                                        :, :guide_text_len
                                    ]
                                    guide_next_tokens_frame_begin_id = calc_next_tokens_frame_begin_id(
                                        guide_text_len, FL, guide_mem_kv_perlayer.shape[1]
                                    )
                                    guide_mems_buffers[id][
                                        layer,
                                        batch_idx : batch_idx + group_size,
                                        guide_text_len : guide_text_len
                                        + guide_mem_kv_perlayer.shape[1]
                                        - guide_next_tokens_frame_begin_id,
                                    ] = guide_mem_kv_perlayer.expand(
                                        min(group_size, input_tokens.shape[0] - batch_idx), -1, -1
                                    )[
                                        :, guide_next_tokens_frame_begin_id:
                                    ]
                                else:
                                    guide_mems_buffers[id][
                                        layer, batch_idx : batch_idx + group_size, : guide_mem_kv_perlayer.shape[1]
                                    ] = guide_mem_kv_perlayer.expand(
                                        min(group_size, input_tokens.shape[0] - batch_idx), -1, -1
                                    )
                        guide_mems_indexs[0], guide_mems_indexs[1] = (
                            guide_mem_kv01[0][0].shape[1],
                            guide_mem_kv01[1][0].shape[1],
                        )
                        if limited_spatial_channel_mem:
                            guide_mems_indexs[0] -= guide_next_tokens_frame_begin_id - guide_text_len
                    guide_mems = [guide_mems_buffers[id][:, :, : guide_mems_indexs[id]] for id in range(2)]
                    guide_logits = guide_logits_all
            else:
                if not mems_buffers_on_GPU:
                    if not mode_stage1:
                        torch.cuda.empty_cache()
                        for idx, mem in enumerate(mems):
                            mems[idx] = mem.to(next(model.parameters()).device)
                        if guide_seq is not None:
                            for idx, mem in enumerate(guide_mems):
                                guide_mems[idx] = mem.to(next(model.parameters()).device)
                        pass
                    else:
                        torch.cuda.empty_cache()
                        for idx, mem_buffer in enumerate(mems_buffers):
                            mems_buffers[idx] = mem_buffer.to(next(model.parameters()).device)
                        mems = [mems_buffers[id][:, :, : mems_indexs[id]] for id in range(2)]
                        if guide_seq is not None:
                            for idx, guide_mem_buffer in enumerate(guide_mems_buffers):
                                guide_mems_buffers[idx] = guide_mem_buffer.to(next(model.parameters()).device)
                            guide_mems = [guide_mems_buffers[id][:, :, : guide_mems_indexs[id]] for id in range(2)]
                        mems_buffers_on_GPU = True

                logits, *output_per_layers = model(
                    input_tokens[:, index:],
                    position_ids[..., index : counter + 1],
                    attention_mask,  # TODO memlen
                    mems=mems,
                    text_len=text_len,
                    frame_len=FL,
                    counter=counter,
                    log_text_attention_weights=log_text_attention_weights,
                    enforce_no_swin=enforce_no_swin,
                    limited_spatial_channel_mem=limited_spatial_channel_mem,
                    **kw_args,
                )
                mem_kv0, mem_kv1 = [o["mem_kv"][0] for o in output_per_layers], [
                    o["mem_kv"][1] for o in output_per_layers
                ]

                if guide_seq is not None:
                    guide_logits, *guide_output_per_layers = model(
                        guide_input_tokens[:, max(index - guide_index_delta, 0) :],
                        guide_position_ids[..., max(index - guide_index_delta, 0) : counter + 1 - guide_index_delta],
                        guide_attention_mask,
                        mems=guide_mems,
                        text_len=guide_text_len,
                        frame_len=FL,
                        counter=counter - guide_index_delta,
                        log_text_attention_weights=0,
                        enforce_no_swin=enforce_no_swin,
                        limited_spatial_channel_mem=limited_spatial_channel_mem,
                        **kw_args,
                    )
                    guide_mem_kv0, guide_mem_kv1 = [o["mem_kv"][0] for o in guide_output_per_layers], [
                        o["mem_kv"][1] for o in guide_output_per_layers
                    ]

                if not mems_buffers_on_GPU:
                    torch.cuda.empty_cache()
                    for idx, mem_buffer in enumerate(mems_buffers):
                        mems_buffers[idx] = mem_buffer.to(next(model.parameters()).device)
                    if guide_seq is not None:
                        for idx, guide_mem_buffer in enumerate(guide_mems_buffers):
                            guide_mems_buffers[idx] = guide_mem_buffer.to(next(model.parameters()).device)
                    mems_buffers_on_GPU = True

                mems, mems_indexs = my_update_mems(
                    [mem_kv0, mem_kv1], mems_buffers, mems_indexs, limited_spatial_channel_mem, text_len, FL
                )
                if guide_seq is not None:
                    guide_mems, guide_mems_indexs = my_update_mems(
                        [guide_mem_kv0, guide_mem_kv1],
                        guide_mems_buffers,
                        guide_mems_indexs,
                        limited_spatial_channel_mem,
                        guide_text_len,
                        FL,
                    )

            counter += 1
            index = counter

            logits = logits[:, -1].expand(batch_size, -1)  # [batch size, vocab size]
            tokens = tokens.expand(batch_size, -1)
            if guide_seq is not None:
                guide_logits = guide_logits[:, -1].expand(batch_size, -1)
                guide_tokens = guide_tokens.expand(batch_size, -1)

            if seq[-1][counter].item() < 0:
                # sampling
                guided_logits = (
                    guide_logits + (logits - guide_logits) * guidance_alpha if guide_seq is not None else logits
                )
                if mode_stage1 and counter < text_len + 400:
                    tokens, mems = strategy.forward(guided_logits, tokens, mems)
                else:
                    tokens, mems = strategy2.forward(guided_logits, tokens, mems)
                if guide_seq is not None:
                    guide_tokens = torch.cat((guide_tokens, tokens[:, -1:]), dim=1)

                if seq[0][counter].item() >= 0:
                    for si in range(seq.shape[0]):
                        if seq[si][counter].item() >= 0:
                            tokens[si, -1] = seq[si, counter]
                            if guide_seq is not None:
                                guide_tokens[si, -1] = guide_seq[si, counter - guide_index_delta]

            else:
                tokens = torch.cat(
                    (
                        tokens,
                        seq[:, counter : counter + 1]
                        .clone()
                        .expand(tokens.shape[0], 1)
                        .to(device=tokens.device, dtype=tokens.dtype),
                    ),
                    dim=1,
                )
                if guide_seq is not None:
                    guide_tokens = torch.cat(
                        (
                            guide_tokens,
                            guide_seq[:, counter - guide_index_delta : counter + 1 - guide_index_delta]
                            .clone()
                            .expand(guide_tokens.shape[0], 1)
                            .to(device=guide_tokens.device, dtype=guide_tokens.dtype),
                        ),
                        dim=1,
                    )

            input_tokens = tokens.clone()
            if guide_seq is not None:
                guide_input_tokens = guide_tokens.clone()
            if (index - text_len - 1) // 400 < (input_tokens.shape[-1] - text_len - 1) // 400:
                boi_idx = ((index - text_len - 1) // 400 + 1) * 400 + text_len
                while boi_idx < input_tokens.shape[-1]:
                    input_tokens[:, boi_idx] = tokenizer["<start_of_image>"]
                    if guide_seq is not None:
                        guide_input_tokens[:, boi_idx - guide_index_delta] = tokenizer["<start_of_image>"]
                    boi_idx += 400

            if strategy.is_done:
                break

            progress.update()

    final_tokens = strategy.finalize(tokens, mems)[0]
    return final_tokens


def process_stage1(
    model,
    seq_text,
    duration,
    use_guidance_stage1,
    device,
    image_prompt,
    strategy_cogview2,
    strategy_cogvideo,
    guidance_alpha,
    keep_mem_buffers,
    num_layers=24,
    hidden_size=1024,
    video_raw_text=None,
    video_guidance_text="视频",
    image_text_suffix="",
):
    model = model.to(device)

    if video_raw_text is None:
        video_raw_text = seq_text

    # generate the first frame:
    enc_text = tokenizer.encode(seq_text + image_text_suffix)
    seq_1st = enc_text + [tokenizer["<start_of_image>"]] + [-1] * 400
    text_len_1st = len(seq_1st) - FL * 1 - 1
    seq_1st = torch.cuda.LongTensor(seq_1st, device=device).unsqueeze(0)
    if image_prompt is None:
        given_tokens = sample_token_sequence(
            model,
            seq_1st.clone(),
            get_masks_and_position_ids=get_masks_and_position_ids_stage1,
            text_len=text_len_1st,
            strategy=strategy_cogview2,
            strategy2=strategy_cogvideo,
            log_text_attention_weights=1.4,
            enforce_no_swin=True,
            mode_stage1=True,
            num_layers=num_layers,
            hidden_size=hidden_size,
            keep_mem_buffers=keep_mem_buffers,
        )[None, None, text_len_1st + 1 : text_len_1st + 401]
    else:
        given_tokens = tokenizer.encode(image_pil=image_prompt, image_size=160).unsqueeze(1)
    # given_tokens.shape: [bs, frame_num, 400]

    # generate subsequent frames:
    enc_duration = tokenizer.encode(str(float(duration)) + "秒")
    if use_guidance_stage1:
        video_raw_text = video_raw_text + " 视频"
    enc_text_video = tokenizer.encode(video_raw_text)
    seq = enc_duration + [tokenizer["<n>"]] + enc_text_video + [tokenizer["<start_of_image>"]] + [-1] * 400 * FN
    guide_seq = (
        enc_duration
        + [tokenizer["<n>"]]
        + tokenizer.encode(video_guidance_text)
        + [tokenizer["<start_of_image>"]]
        + [-1] * 400 * FN
    )

    text_len = len(seq) - FL * FN - 1
    guide_text_len = len(guide_seq) - FL * FN - 1
    seq = torch.cuda.LongTensor(seq, device=device).unsqueeze(0)
    guide_seq = torch.cuda.LongTensor(guide_seq, device=device).unsqueeze(0)

    for given_frame_id in range(given_tokens.shape[1]):
        seq[:, text_len + 1 + given_frame_id * 400 : text_len + 1 + (given_frame_id + 1) * 400] = given_tokens[
            :, given_frame_id
        ]
        guide_seq[
            :, guide_text_len + 1 + given_frame_id * 400 : guide_text_len + 1 + (given_frame_id + 1) * 400
        ] = given_tokens[:, given_frame_id]

    if use_guidance_stage1:
        video_log_text_attention_weights = 0
    else:
        guide_seq = None
        video_log_text_attention_weights = 1.4

    output_tokens = sample_token_sequence(
        model,
        seq.clone(),
        get_masks_and_position_ids=get_masks_and_position_ids_stage1,
        text_len=text_len,
        strategy=strategy_cogview2,
        strategy2=strategy_cogvideo,
        log_text_attention_weights=video_log_text_attention_weights,
        guide_seq=guide_seq.clone() if guide_seq is not None else None,
        guide_text_len=guide_text_len,
        guidance_alpha=guidance_alpha,
        limited_spatial_channel_mem=True,
        mode_stage1=True,
        num_layers=num_layers,
        hidden_size=hidden_size,
        keep_mem_buffers=keep_mem_buffers,
    )[:, 1 + text_len :]

    model = model.cpu()
    torch.cuda.empty_cache()

    imgs = [
        torch.nn.functional.interpolate(
            tokenizer.decode(image_ids=output_tokens.squeeze().tolist()[i * FL : (i + 1) * FL]), size=(480, 480)
        )
        for i in range(FN)
    ]
    save_tokens = output_tokens[:, : FN * FL].reshape(-1, FN, FL).cpu()
    return imgs, save_tokens


def process_stage2(
    model,
    seq_text,
    parent_tokens,
    duration,
    use_guidance_stage2,
    device,
    strategy_cogview2,
    strategy_cogvideo,
    guidance_alpha,
    keep_mem_buffers,
    dsr,
    num_layers=48,
    hidden_size=3072,
    video_guidance_text="视频",
):
    model = model.cuda()

    # CogVideo Stage2 Generation
    while duration >= 0.5:  # TODO: You can change the boundary to change the frame rate
        enc_text = tokenizer.encode(seq_text)
        enc_duration = tokenizer.encode(str(float(duration)) + "秒")
        seq = enc_duration + [tokenizer["<n>"]] + enc_text + [tokenizer["<start_of_image>"]] + [-1] * 400 * FN
        text_len = len(seq) - FL * FN - 1
        seq[text_len + 1 : text_len + 1 + 400] = parent_tokens[0][0]
        seq[text_len + 1 + 400 : text_len + 1 + 800] = parent_tokens[0][1]
        seq[text_len + 1 + 800 : text_len + 1 + 1200] = parent_tokens[0][2]
        seq = torch.cuda.LongTensor(seq, device=device).unsqueeze(0)

        if use_guidance_stage2:
            guide_seq = (
                enc_duration
                + [tokenizer["<n>"]]
                + tokenizer.encode(video_guidance_text)
                + [tokenizer["<start_of_image>"]]
                + [-1] * 400 * FN
            )
            guide_text_len = len(guide_seq) - FL * FN - 1
            guide_seq[guide_text_len + 1 : guide_text_len + 1 + 400] = parent_tokens[0][0]
            guide_seq[guide_text_len + 1 + 400 : guide_text_len + 1 + 800] = parent_tokens[0][1]
            guide_seq[guide_text_len + 1 + 800 : guide_text_len + 1 + 1200] = parent_tokens[0][2]
            guide_seq = torch.cuda.LongTensor(guide_seq, device=device).unsqueeze(0)
            video_log_text_attention_weights = 0
        else:
            guide_seq = None
            guide_text_len = 0
            video_log_text_attention_weights = 1.4

        print("myfillseq", duration)
        output_tokens = sample_token_sequence(
            model,
            seq.clone(),
            get_masks_and_position_ids=get_masks_and_position_ids_stage2,
            text_len=text_len,
            strategy=strategy_cogview2,
            strategy2=strategy_cogvideo,
            log_text_attention_weights=video_log_text_attention_weights,
            mode_stage1=False,
            guide_seq=guide_seq.clone() if guide_seq is not None else None,
            guide_text_len=guide_text_len,
            guidance_alpha=guidance_alpha,
            limited_spatial_channel_mem=True,
            num_layers=num_layers,
            hidden_size=hidden_size,
            keep_mem_buffers=keep_mem_buffers,
        )
        output_tokens = output_tokens[:, text_len + 1 : text_len + 1 + FN * 400].reshape(1, -1, 400 * FN)
        output_tokens_merge = torch.cat(
            (
                output_tokens[:, :, : 1 * 400],
                output_tokens[:, :, 400 * 3 : 4 * 400],
                output_tokens[:, :, 400 * 1 : 2 * 400],
                output_tokens[:, :, 400 * 4 : FN * 400],
            ),
            dim=2,
        ).reshape(1, -1, 400)
        output_tokens_merge = torch.cat((output_tokens_merge, output_tokens[:, -1:, 400 * 2 : 3 * 400]), dim=1)
        duration /= 2
        parent_tokens = output_tokens_merge

    model = model.cpu()
    torch.cuda.empty_cache()

    # decoding
    if keep_mem_buffers:
        del mem_dict["guide_buffer"]
        del mem_dict["buffer"]
        mem_dict["guide_buffer"] = None
        mem_dict["buffer"] = None

    # direct super-resolution by CogView2
    enc_text = tokenizer.encode(seq_text)
    frame_num_per_sample = parent_tokens.shape[1]
    parent_tokens_2d = parent_tokens.reshape(-1, 400)
    text_seq = torch.cuda.LongTensor(enc_text, device=device).unsqueeze(0).repeat(parent_tokens_2d.shape[0], 1)
    sred_tokens = dsr(text_seq, parent_tokens_2d)
    print(len(sred_tokens), len(sred_tokens[0]))

    decoded_sr_imgs = []
    for frame_i in range(frame_num_per_sample):
        decoded_sr_img = tokenizer.decode(image_ids=sred_tokens[frame_i + frame_num_per_sample][-3600:])
        decoded_sr_imgs.append(torch.nn.functional.interpolate(decoded_sr_img, size=(480, 480)))
    return decoded_sr_imgs


class InferenceModel_Sequential(CogVideoCacheModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(
            args, transformer=transformer, parallel_output=parallel_output, window_size=-1, cogvideo_stage=1
        )

    def final_forward(self, logits, **kwargs):
        logits_parallel = logits
        logits_parallel = torch.nn.functional.linear(
            logits_parallel.float(), self.transformer.word_embeddings.weight[:20000].float()
        )
        return logits_parallel


class InferenceModel_Interpolate(CogVideoCacheModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(
            args, transformer=transformer, parallel_output=parallel_output, window_size=10, cogvideo_stage=2
        )

    def final_forward(self, logits, **kwargs):
        logits_parallel = logits
        logits_parallel = torch.nn.functional.linear(
            logits_parallel.float(), self.transformer.word_embeddings.weight[:20000].float()
        )
        return logits_parallel


SHARED_CONF = dict(
    additional_seqlen=2000,
    hidden_dropout=0.1,
    attention_dropout=0.1,
    inner_hidden_size=None,
    hidden_size_per_attention_head=None,
    checkpoint_activations=False,
    checkpoint_num_layers=1,
    skip_init=True,
    use_gpu_initialization=False,
    fp16=True,
    mode="inference",
)
SEQ_CONF = argparse.Namespace(
    num_layers=24,
    vocab_size=0,
    hidden_size=1024,
    num_attention_heads=16,
    max_sequence_length=512,
    layernorm_order="pre",
    **SHARED_CONF,
)
BIG_CONF = dict(
    num_layers=48,
    vocab_size=150010,
    hidden_size=3072,
    num_attention_heads=48,
    max_sequence_length=1024,
    layernorm_order="sandwich",
    **SHARED_CONF,
)


def main(
    output_path,
    text,
    translate,
    image_prompt,
    input_dir,
    number,
    use_guidance_stage1,
    use_guidance_stage2,
    guidance_alpha,
    temperature,
    coglm_temperature2,
    top_k,
    stage_1,
    stage_2,
    both_stages,
    keep_mem_buffers,
    device,
):
    assert int(stage_1) + int(stage_2) + int(both_stages) == 1
    device = torch.device(device)

    if translate:
        translator = GoogleTranslator(source="en", target="zh-CN")
        text = translator.translate(text)

    if stage_1 or both_stages:
        model_stage1, _ = InferenceModel_Sequential.from_pretrained(args=SEQ_CONF, name="cogvideo-stage1")
        model_stage1.eval()
        if both_stages:
            model_stage1 = model_stage1.cpu()

    if stage_2 or both_stages:
        model_stage2, _ = InferenceModel_Interpolate.from_pretrained(
            args=argparse.Namespace(model_class="CogVideoModel", **BIG_CONF),
            name="cogvideo-stage2",
        )
        model_stage2.eval()
        if both_stages:
            model_stage2 = model_stage2.cpu()

    if not stage_1:
        dsr_path = auto_create("cogview2-dsr", path=None)
        dsr = DirectSuperResolution(argparse.Namespace(**BIG_CONF), dsr_path, max_bz=1, onCUDA=False)

    torch.cuda.empty_cache()
    invalid_slices = [slice(tokenizer.num_image_tokens, None)]
    strategy_cogview2 = CoglmStrategy(invalid_slices, temperature=1.0, top_k=16)
    strategy_cogvideo = CoglmStrategy(
        invalid_slices, temperature=temperature, top_k=top_k, temperature2=coglm_temperature2
    )

    for n in range(number):
        if keep_mem_buffers:
            torch.cuda.empty_cache()
            tweak_mems_buffers = [
                torch.zeros(48, 1, mem_len, 3072 * 2, dtype=next(model_stage2.parameters()).dtype, device=device)
                for mem_len in [400 + 74, 5 * 400 + 74]
            ]
            mem_dict["buffer"] = tweak_mems_buffers
            if use_guidance_stage1:
                tweak_guide_mems_buffers = deepcopy(tweak_mems_buffers)
                mem_dict["guide_buffer"] = tweak_guide_mems_buffers

        if stage_1 or both_stages:

            if input_dir is not None:
                image_prompt = random.choice(glob(f"{input_dir}/*"))

            path = os.path.join(output_path, f"{Path(image_prompt).stem}_{text}")

            imgs, tokens = process_stage1(
                model=model_stage1,
                seq_text=text,
                duration=4.0,
                use_guidance_stage1=use_guidance_stage1,
                device=device,
                num_layers=24,
                hidden_size=1024,
                image_prompt=Image.open(image_prompt).convert("RGB"),
                keep_mem_buffers=keep_mem_buffers,
                strategy_cogview2=strategy_cogview2,
                strategy_cogvideo=strategy_cogvideo,
                guidance_alpha=guidance_alpha,
                video_raw_text=text,
                video_guidance_text="视频",
                image_text_suffix=" 高清摄影",
            )

            out_dir = path + "_stage1"
            my_save_multiple_images(imgs, out_dir, subdir=f"frames/0", debug=False)
            os.system(f"gifmaker -i '{out_dir}'/frames/0/0*.jpg -o '{out_dir}/0.gif' -d 0.25")
            torch.save(tokens, os.path.join(out_dir, "frame_tokens.pt"))

            if both_stages:
                imgs = process_stage2(
                    model=model_stage2,
                    dsr=dsr,
                    seq_text=text,
                    parent_tokens=tokens,
                    duration=2.0,
                    use_guidance_stage2=use_guidance_stage2,
                    device=device,
                    num_layers=48,
                    hidden_size=3072,
                    strategy_cogview2=strategy_cogview2,
                    strategy_cogvideo=strategy_cogvideo,
                    guidance_alpha=guidance_alpha,
                    keep_mem_buffers=keep_mem_buffers,
                    video_guidance_text="视频",
                )

                out_dir = path + "_stage2"
                my_save_multiple_images(imgs, out_dir, subdir=f"frames/0", debug=False)
                os.system(f"gifmaker -i '{out_dir}'/frames/0/0*.jpg -o '{out_dir}/0.gif' -d 0.125")

        elif stage_2:
            sample_dirs = os.listdir(output_path)
            for sample in sample_dirs:
                text = sample.split("_")[-1]
                path = os.path.join(output_path, sample, "Interp")
                parent_tokens = torch.load(os.path.join(output_path, sample, "frame_tokens.pt"))
                imgs = process_stage2(
                    model=model_stage2,
                    dsr=dsr,
                    seq_text=text,
                    parent_tokens=parent_tokens,
                    duration=2.0,
                    use_guidance_stage2=use_guidance_stage2,
                    device=device,
                    num_layers=48,
                    hidden_size=3072,
                    strategy_cogview2=strategy_cogview2,
                    strategy_cogvideo=strategy_cogvideo,
                    guidance_alpha=guidance_alpha,
                    keep_mem_buffers=keep_mem_buffers,
                    video_guidance_text="视频",
                )
                out_dir = path + "_stage2"
                my_save_multiple_images(imgs, out_dir, subdir=f"frames/0", debug=False)
                os.system(f"gifmaker -i '{out_dir}'/frames/0/0*.jpg -o '{out_dir}/0.gif' -d 0.125")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)

    parser.add_argument("--number", type=int, default=1)

    parser.add_argument("--output-path", type=str, default="output/")

    parser.add_argument("--keep-mem-buffers", action="store_true")

    parser.add_argument("--translate", action="store_true")

    parser.add_argument("--use-guidance-stage1", action="store_true")
    parser.add_argument("--use-guidance-stage2", action="store_true")
    parser.add_argument("--guidance-alpha", type=float, default=3.0)

    parser.add_argument("--stage-1", action="store_true")
    parser.add_argument("--stage-2", action="store_true")
    parser.add_argument("--both-stages", action="store_true")

    parser.add_argument("--coglm-temperature2", type=float, default=0.89)
    parser.add_argument("--temperature", type=float, default=1.05)
    parser.add_argument("--top-k", type=int, default=12)

    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--model-parallel-size", type=int, default=1)

    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    init_method = "tcp://"
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    if args.device == -1:  # not set manually
        args.device = args.rank % torch.cuda.device_count()
    args.master_ip = os.getenv("MASTER_ADDR", "localhost")
    args.master_port = os.getenv("MASTER_PORT", "6000")
    init_method += args.master_ip + ":" + args.master_port
    torch.distributed.init_process_group(
        backend="nccl", world_size=args.world_size, rank=args.rank, init_method=init_method
    )
    mpu.initialize_model_parallel(args.model_parallel_size)

    with torch.inference_mode():
        main(
            output_path=args.output_path,
            text=args.text,
            translate=args.translate,
            image_prompt=args.image,
            input_dir=args.input_dir,
            number=args.number,
            use_guidance_stage1=args.use_guidance_stage1,
            use_guidance_stage2=args.use_guidance_stage2,
            guidance_alpha=args.guidance_alpha,
            temperature=args.temperature,
            coglm_temperature2=args.coglm_temperature2,
            top_k=args.top_k,
            stage_1=args.stage_1,
            stage_2=args.stage_2,
            both_stages=args.both_stages,
            keep_mem_buffers=args.keep_mem_buffers,
            device=args.device,
        )
