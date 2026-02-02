from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import log_softmax, nll_loss


PROMPT_DICT = {
    "prompt_input": "USER:\n{instruction}\n\n{input}\n\nASSISTANT:\n",
    "prompt_no_input": "USER:\n{instruction}\n\nASSISTANT:\n"
}

def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    start_index = text.rfind(target_span)
    start_token = len(tokenizer.encode(text[:start_index]))
    end_token = input_ids.shape[1]
    
    labels = input_ids.clone()
    labels[0, :start_token] = -100
    
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
    
    loss = outputs.loss
    perplexity = torch.exp(loss)
    
    losses = []
    logits = outputs.logits
    for i in range(1, end_token):
        log_prob_dist = log_softmax(logits[0, i-1])
        true_token = input_ids[0, i]
        token_loss = nll_loss(log_prob_dist.unsqueeze(0), true_token.unsqueeze(0))
        losses.append(token_loss.item())
    
    return loss, perplexity, losses

def compute_gradnorm_reward(model, sample):
    lst = []
    model.eval()
    for example in sample:
        model.zero_grad()
        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]
        labels = input_ids.clone().detach().to(model.device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lst.append(grad_norm.item())
    return sum(lst) / len(lst)

def calculate_loss_per_example(example, tokenizer, model, max_length=4096):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_i = example

    instruct_i = data_i['instruction']
    output_i = data_i['output']

    direct_answer_text = "ASSISTANT:\n"+output_i
    whole_text = PROMPT_DICT["prompt_no_input"].format_map({'instruction': instruct_i})+output_i

    input_i = data_i.get('input', '')
    if input_i:
        # whole_text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruct_i}\n\n### Input:\n{input_i}\n\n### Response: {output_i}"
        whole_text = PROMPT_DICT["prompt_input"].format_map({'instruction': instruct_i, 'input': input_i})+output_i
    
    instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    instruct_i_len = instruct_i_input_ids.shape[1]
    # loss_out_alone, ppl_out_alone, loss_list_alone = get_perplexity_and_embedding_part_text(tokenizer, model, direct_answer_text, output_i, max_length-instruct_i_len+4)
    loss_out_condition, ppl_out_condition, loss_list_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, output_i, max_length)

    return loss_out_condition

data = load_dataset("bigstupidhats/dynasample_train_scoreby3llms", split="Chinese")

model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")

example = data[0]
print(example["Qwen2.5-7B_loss"], example["Qwen2.5-7B_ppl"])
print(calculate_loss_per_example(example, tokenizer, model))

# sample = []
# for i in range(3):
#     text = tokenizer.apply_chat_template(data["English"][i]["conversations"], tokenize=False)
#     sample.append(tokenizer(text, return_tensors="pt"))

# print(compute_gradnorm_reward(model, sample))
