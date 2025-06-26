from tqdm.notebook import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model= AutoModelForCausalLM.from_pretrained('/gpt2-dialogue-full')
tokenizer = AutoTokenizer.from_pretrained('/gpt2-dialogue-full')

def generate_text(prompt,max_t=10,temp=1.0,top_k=50,top_p=0.9):
  device = model.device
  input_ids = tokenizer.encode(prompt,return_tensors="pt").to(device)
  # print(input_ids.shape)
  gen_text = ""

  total_loss = 0
  num_tokens = 0

  for _ in range(max_t):
    with torch.no_grad():
      outputs = model(input_ids,labels = input_ids)#logits have (B,Seq_len,Vocab_size)
      loss = outputs.loss
      total_loss += loss.item()
      num_tokens += 1
      logits = outputs.logits[:,-1,:] # here we get the last one bc we dont count others
      logits = logits/temp

      if top_k is not None:
        top_k = min(top_k,logits.size(-1))# we need this not to go over the size of logits
        indices_to_remove = logits< torch.topk(logits,top_k)[0][...,-1,None] #[0] - is values of topk  This is the last value bc of -1
        logits[indices_to_remove] = -float("inf")

      if top_p is not None and 0<top_p <1:
        sorted_vals , sorted_indices = torch.sort(logits,descending=True)
        fixed_probs = torch.softmax(sorted_vals,dim=-1).cumsum(dim=-1)

        indices_to_removes = fixed_probs > top_p
        indices_to_removes[...,1:] = indices_to_removes[...,:-1].clone()
        indices_to_removes[...,0] = False

        indices_to_remove = indices_to_removes.scatter(dim=1,index=sorted_indices,src=indices_to_removes)
        logits[indices_to_remove] = -float('inf')

      probs = torch.softmax(logits,dim=-1)
      new_token = torch.multinomial(probs,num_samples=1)

    input_ids = torch.cat([input_ids,new_token],dim=1)
    new_word = tokenizer.decode(new_token[0])
    gen_text += new_word
    print(f"\r{gen_text}", end='', flush=True)
  response = str(tokenizer.decode(input_ids[0]))  # bc input_ids has shape [1,14]
  meta_response = response
  response = "".join(list(response)[len(prompt):])
  perplexity = torch.exp(torch.tensor(total_loss/num_tokens)).item()
  return response,perplexity,meta_response

trig = True
memory = None
print("Type exit to stop the machine")
while trig == True:
  print()
  prompt = input()
  if prompt =="exit":
    trig = False
    break
  if memory is not None:
    prompt = memory + f'<USER>: {prompt}<BOT>:'
  else:
    prompt = f'<USER>: {prompt}<BOT>:'
  res,_,meta = generate_text(prompt)