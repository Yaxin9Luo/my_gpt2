import os
import torch
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# GPT-2 模型和相关配置的导入
from gpt2.models.model import GPTConfig, GPT
import tiktoken
# FastAPI 应用实例
app = FastAPI()
# 模型初始化和配置
init_from = 'resume'  # 'resume' 或 GPT-2 变体（如 'gpt2-xl'）
out_dir = 'out-shakespeare-char'  # 如果 init_from 不是 'resume' 则忽略
start = "\n"  # 起始文本，可以是 "" 或其他。也可以指定文件，使用方式为 "FILE:prompt.txt"
num_samples = 10  # 生成样本数量
max_new_tokens = 500  # 每个样本中生成的最大token数量
temperature = 0.8  # 温度，用于影响随机性
top_k = 200  # 仅保留 top_k 最有可能的tokens
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备类型，如 'cpu', 'cuda', 'cuda:0'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
# 设置随机种子和其他配置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type != 'cpu' else torch.no_grad()

# 模型初始化
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile and torch.version >= '2.0':
    model = torch.compile(model)

# 加载编码器和解码器
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={""})
decode = lambda l: enc.decode(l)



# 请求体定义
class GenerationRequest(BaseModel):
    input_text: str

# 路由定义
@app.get("/generate/")
async def generate_text():
    # 生成文本
    generated_text = ""
    with torch.no_grad():
        with ctx:
            # 使用模型直接生成文本
            y = model.generate(max_new_tokens, temperature=temperature, top_k=top_k)
            generated_text = decode(y.tolist())

    return {"generated_text": generated_text}

# 如果直接运行此脚本，则启动 FastAPI 服务器
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)