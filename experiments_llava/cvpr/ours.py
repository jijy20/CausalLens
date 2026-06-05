import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import numpy as np
from PIL import Image
import math
import pdb
# import kornia
from transformers import set_seed
from causallens_utils.vcd_add_noise import add_diffusion_noise, pad_to_square
from causallens_utils.sample_causallens import evolve_ours_sampling
from AttnAdapter import AttnAdapter
evolve_ours_sampling()



def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    for i, layer in enumerate(model.model.layers):
        if i >= args.layer_start and i <= args.layer_end:
            # attn_adap = AttnAdapter(layer.self_attn.config)
            attn_adap = AttnAdapter(
                layer.self_attn.config,
                lambda_causal=args.lambda_causal,
                gamma_mix=args.gamma_mix,
                sys_len=args.sys_len,
                img_len=args.img_len
            )
            attn_adap.load_state_dict(layer.self_attn.state_dict())
            attn_adap = attn_adap.half().cuda()
            layer.self_attn = attn_adap
    

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word. ")
        # conv.append_message(conv.roles[0], qs )
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print('prompt',prompt)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image = pad_to_square(image)

# Add these two lines to ensure it's a standard RGB PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image).astype(np.uint8))
        image = image.convert('RGB')

        processed = image_processor.preprocess(image)  # without return_tensors argument
        pixel_values = np.array(processed['pixel_values'][0], dtype=np.float32)
        image_tensor = torch.from_numpy(np.ascontiguousarray(pixel_values))

        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None      

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            model.tokenizer = tokenizer
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True)
            # print('output_ids',output_ids.shape)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/checkpoint/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/dataset/val/val2014")
    parser.add_argument("--question-file", type=str, default="/experiments/data/POPE/coco/coco_pope_random.json")
    parser.add_argument("--answers-file", type=str, default="/experiments_llava/cvpr/cocorandomllava1.5my_0.15_10_20.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=999)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--lambda_causal", type=float, default=0.15, help="Causal intervention strength")
    parser.add_argument("--gamma_mix", type=float, default=0.15, help="Mixing ratio between residual and replacement")
    parser.add_argument("--sys_len", type=int, default=35, help="System token count")
    parser.add_argument("--img_len", type=int, default=576, help="Image token count")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--layer_start", type=int, default=10, help="Start layer index (inclusive)")
    parser.add_argument("--layer_end", type=int, default=20, help="End layer index (inclusive)")
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
