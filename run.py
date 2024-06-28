from llava.eval.run_llava import eval_model
import time
model_path = "liuhaotian/llava-v1.5-7b"
prompt = 'Can you write me something about Eminem?'
#image_file = "./playground/data/C1-TM20240411-124417.17856-B3-V0-T0-5565-1-CNT3235-PIC1_C3N.png"
image_file = "./playground/data/canjiao.png"
#prompt = '/home/main/anaconda3/envs/llava/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:392: UserWarning: do_sample is set to False. However, temperature is set to 0 -- this flag is only used in sample-based generation modes. You should set do_sample=True or unset temperature.'

# args = type('Args', (), {
#     "model_path": "./checkpoints/llava-v1.5-7b",
#     "model_base": None,
#     "model_name": "liuhaotian/llava-v1.5-7b",
#     "query": prompt,
#     "conv_mode": None,
#     "image_file": image_file,
#     "sep": ",",
#     "temperature": 0,
#     "top_p": None,
#     "num_beams": 1,
#     "max_new_tokens": 512
# })()

print("原始模型输出为：")
#eval_model(args)

args = type('Args', (), {
    "model_path": "./checkpoints/llava-v1.5-7b-merged_less",
    "model_base": None,
    "model_name": "liuhaotian/llava-v1.5-7b",
    "query": prompt,
    "conv_mode": None,
    "image_file": None,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()
#eval_model(args)
t0 = time.time()
print("微调后的模型输出为：")
eval_model(args)
t1 = time.time()
print(t1-t0)
