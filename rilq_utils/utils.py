







import logging

def get_logger(
    logging_path = None,
    logging_name = "noName"
):
    log_format = '[%(levelname)s|%(funcName)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s'

    logger = logging.getLogger("transformers")
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    if logging_path:
        file_handler = logging.FileHandler(logging_path)
    stream_handler = logging.StreamHandler()
    
    formatter = logging.Formatter(log_format)
    
    stream_handler.setFormatter(formatter)
    if logging_path:
        file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if logging_path:
        logger.addHandler(file_handler)
    return logger



from torch.nn import CrossEntropyLoss


# from modeling_llama.py
def get_gt_loss(model, labels, input_tensor, is_logit = False):
    '''
    if model.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(model.vocab_size // model.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(model.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = model.lm_head(hidden_states)
    '''
    if not is_logit:
        logits = model.lm_head(input_tensor)
        logits = logits.float()
    else:
        logits = input_tensor.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
    return loss



MODEL_INFO={
    "Llama-2-7b-hf":{
        'linears': {'q_proj','k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj'},
        'sublayers': {'mlp','self_attn'}
    },
    "Llama-2-13b-hf":{
        'linears': {'q_proj','k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj'},
        'sublayers': {'mlp','self_attn'}
    },
    "Llama-2-70b-hf":{
        'linears': {'q_proj','k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj'},
        'sublayers': {'mlp','self_attn'}
    },
    "Meta-Llama-3-8B":{
        'linears': {'q_proj','k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj'},
        'sublayers': {'mlp','self_attn'}
    },

    "roberta-large":{
        'linears': {'query','key','value','output.dense','intermediate.dense'}
    }
}



def get_target_modules(module, targets):    
    return {name: m for name, m in module.named_modules() if name.split('.')[-1] in targets}



from os.path import join

def dump_args(args):
    with open(join(args.output_dir, 'args.txt'),'w') as f_w:
        for k,v in sorted(vars(args).items()):
            f_w.write(f'{k}\t{v}\n')




import torch.nn as nn


class Shell(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)


def unwrap_model(model, sub_module_name=".base_layer"):
    sub_module_name_list = [k.split(sub_module_name)[0] for k in model.state_dict().keys() if sub_module_name in k]
    sub_module_name_set = set(sub_module_name_list)
    for name in sub_module_name_set:
        # get the parent of the submodule
        name_parent = ".".join(name.split(".")[:-1])
        name_child = name.split(".")[-1]
        sub_module = model.get_submodule(name_parent)
        #print(sub_module)

        # replace with shell
        child = getattr(sub_module, name_child)
        weight = getattr(child.base_layer, "weight", None)
        bias = getattr(child.base_layer, "bias", None)
        shell = Shell(weight, bias)

        setattr(sub_module, name_child, shell)

    print("You have unwrapped the model. Use it on your own risk.")


import json

def dump_jsonl(path, target):
    with open(path, 'w') as f_w:
        tot = len(target)
        for i, each_log in enumerate(target):
            json.dump(each_log, f_w)
            if i != tot-1:
                f_w.write('\n')

