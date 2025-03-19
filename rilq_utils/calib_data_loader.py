import copy
import torch
import datasets
from datasets import load_from_disk, load_dataset
#from accelerate.utils import set_seed
from torch.nn.utils.rnn import pad_sequence
#from optimum.gptq.data import get_dataset, prepare_dataset
from rilq_utils.data import get_dataset, prepare_dataset

import random 
ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}



def get_calib_dataset(
    calib_datasets, 
    calib_num_samples, 
    tokenizer, 
    format_type, 
    val_ratio = None, 
    seed = 42, 
    logger = None, 
    calib_max_length = None
):
    random.seed(seed)
    assert type(calib_datasets) == list and type(calib_num_samples) == list
    assert len(calib_datasets) == len(calib_num_samples)
    #assert format_type =='approx' and val_ratio != None
    
    tot = []
    v_tot = []
    target_length = None
    for calib_dataset, calib_num_sample in zip(calib_datasets, calib_num_samples):
        if calib_num_sample==0: continue
        if calib_dataset == 'c4' or calib_dataset == 'wikitext2':
            each_dset = get_ptl_dataset(calib_dataset, calib_num_sample, calib_max_length, tokenizer, format_type, seed = seed)
        else:
            raise Exception(f"{calib_dataset} - Not Implemented")
            
    
        if val_ratio: 
            val_num = int(calib_num_sample * val_ratio)
            v_tot.extend(each_dset[-val_num:])
            tot.extend(each_dset[:-val_num])
        else:
            tot.extend(each_dset)

    if val_ratio:
        random.shuffle(tot)
        random.shuffle(v_tot)
        pivot = len(v_tot)
        tot = v_tot + tot
    else:
        random.shuffle(tot)
        pivot=0
    
    if format_type == 'approx':
        
        p_inputs, p_masks = pad_and_cut(
            tot, 
            target_length = calib_max_length if calib_max_length != None else target_length, 
            pad_token_id = tokenizer.pad_token_id
        )
        #logger.info(str(p_inputs[0]))
       
        
        return p_inputs, p_masks, pivot
    
    elif format_type == 'gptq':
        tot = tot[pivot:]
        logger.info(f'number of samples: {len(tot)}')
        logger.info(str(tot[0]))
        return tot


    
def get_ptl_dataset(
    calib_dataset,
    calib_num_sample,
    calib_max_length,
    tokenizer,
    format_type,
    seed = 42
):
    if not calib_max_length:
        calib_max_length=2048
    print(calib_num_sample)
    each_dset = get_dataset(calib_dataset, tokenizer, calib_num_sample, seed = seed, seqlen = calib_max_length)

    return each_dset



def pad_and_cut(
    samples,
    target_length,
    pad_token_id
):
    new_input_ids = []
    new_att_masks = []
    for sample in samples:
        input_ids = sample['input_ids']
        att_mask = sample['attention_mask']
        sample_len = len(sample['attention_mask'][0])
        '''
        print(input_ids)
        print(att_mask)
        exit()
        '''

        if sample_len < target_length:
            pad_len = target_length - sample_len
            pad_input = torch.ones(pad_len, dtype = input_ids.dtype) * pad_token_id
            pad_att_mask = torch.zeros(pad_len, dtype = att_mask.dtype)

            new_input_ids.append(torch.cat((input_ids[0], pad_input)))
            new_att_masks.append(torch.cat((att_mask[0], pad_att_mask)))

        elif target_length < sample_len:
            new_input_ids.append(sample['input_ids'][0][:target_length])
            new_att_masks.append(sample['attention_mask'][0][:target_length])
        
        else:
            new_input_ids.append(input_ids[0])
            new_att_masks.append(att_mask[0])
    
    return torch.stack(new_input_ids), torch.stack(new_att_masks)

        

        



