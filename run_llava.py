import time
import os
import torch
from transformers import GenerationConfig
from llava.eval.run_llava import (
    load_pretrained_model,
    tokenizer_image_token,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    conv_templates,
    load_images,
    process_images,
    image_parser,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
def main():
    # 模型加载和初始化
    model_path = "./checkpoints/llava-v1.5-7b-merged"
    #model_path = "./checkpoints/llava-v1.6-vicuna-7b"
    model_base = None
    model_name = "liuhaotian/llava-v1.5-7b"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,model_base,model_name)
    model.cuda().eval()

    # 上下文处理
    model_name = os.path.basename(model_path)
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()

    while True:
        # 获取用户输入的文本
        prompt = input("请输入文本（输入 'exit' 退出）: ")
        if prompt.lower() == 'exit':
            break

        # 询问用户是否提供图片
        image_file = None
        while True:
            image_choice = input("是否提供图片？(yes/no): ")
            if image_choice.lower() == 'yes':
                image_file = input("请输入图片文件路径: ")
                print(image_file)
                if not os.path.exists(image_file):
                    print("图片文件不存在，请重新输入。")
                    continue
                print("图片存在")
                image_files = [image_file]
                break
            elif image_choice.lower() == 'no':
                break
            else:
                image_file = None
                print("无效的输入，请输入 'yes' 或 'no'。")
                
        t0 = time.time()
        # 上下文处理
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        image_info = None

        # 图片处理
        if image_file is not None:
            #image_files = image_parser(image_file)
            images = load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                    images,
                    image_processor,
                    model.config
                ).to(model.device, dtype=torch.float16)
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            image_info = (images_tensor,image_sizes)
            if IMAGE_PLACEHOLDER in prompt:
                if model.config.mm_use_im_start_end:
                    prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, prompt)
                else:
                    prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt)
            else:
                if model.config.mm_use_im_start_end:
                    prompt = image_token_se + "\n" + prompt
                else:
                    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        else:
            if image_info is not None:
                images_tensor,image_sizes = image_info
            else:
                images_tensor = None
                image_sizes = None


        # 生成输出
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).cuda()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        t1 = time.time()
        print(outputs)
        print("模型运行时间：",t1-t0)

        # 询问用户是否重新开始对话
        restart_choice = input("是否继续对话？no的话将会清空上下文。(yes/no): ")
        if restart_choice.lower() != 'no':
            continue

        # 重置对话上下文
        conv = conv_templates[conv_mode].copy()

if __name__ == "__main__":
    main()
