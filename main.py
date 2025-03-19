import os, json, sys, math
import gc
from os.path import join
import argparse
from shutil import rmtree
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from safetensors import safe_open

import ast
import torch.optim as optim
from accelerate.utils import set_seed

from rilq_utils.calib_data_loader import get_calib_dataset

from rilq_utils.utils import (
    get_gt_loss, 
    get_logger, 
    MODEL_INFO, 
    get_target_modules,
    dump_jsonl,
    unwrap_model,
    dump_args
)


if __name__ == '__main__':

    #===================================================================================================
    # Step 1: Preparation 
    #===================================================================================================

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type = str) # FP_Model 
    parser.add_argument("--q_model", type = str)    # Q_Model (Fake Quantized)
    parser.add_argument("--lora_r", type = int, default=64)
    parser.add_argument("--lora_alpha", type = int, default=16)
    parser.add_argument("--lora_dropout", type = float, default= 0.0)

    parser.add_argument("--calib_datasets", type = str) # "c4" or "wikitext2"
    parser.add_argument("--calib_num_samples", type = str)
    parser.add_argument("--calib_val_ratio", type = float)
    parser.add_argument("--calib_max_length", type = int, default = None)

    parser.add_argument("--a_type", default = 'hid') # "hid" or "logit"
    parser.add_argument("--a_dis_weight", type = float, default= 0.5)
    parser.add_argument("--gt_weight", type = float, default = 0.5)

    parser.add_argument("--approx_batch_size", type = int, default=64)
    parser.add_argument("--approx_lr", type = float, default=0.0001)
    parser.add_argument("--approx_total_steps", type = int, default=10000)
    parser.add_argument("--approx_eval_steps", type = int, default=25)
    parser.add_argument("--approx_early_stop", action = 'store_true')
    parser.add_argument("--approx_es_delta", type = float, default = 0.0)
    parser.add_argument("--approx_es_patience", type = int, default = 5)    
    parser.add_argument("--gradient_accumulation_steps", type = int, default = 1)

    parser.add_argument("--m_seed", type = int, default=42)
    parser.add_argument("--output_dir", type = str, default = 'test')
    parser.add_argument("--dtype", type = str)

    args = parser.parse_args()

    set_seed(args.m_seed)
    if os.path.isdir(args.output_dir):
        raise Exception('Output Directory is already taken') 
    os.mkdir(args.output_dir)
    os.mkdir(join(args.output_dir, 'log'))

    logger = get_logger(
        logging_path = join(args.output_dir, 'log.log')
    )    

    logger.info(args)

    if args.dtype:
        if args.dtype == 'bf16':
            dtype = torch.bfloat16
        elif args.dtype == 'fp16':
            dtype = torch.float16
        else:
            raise Exception(f'{args.dtype} Data Type - Not Implemented') 
    else:
        dtype = torch.float32    

    #=========================================================================================
    # Step 2: Get FP Activation
    #=========================================================================================

    s_time = time.time()

    # Step 2-1: Load FP Model
    logger.info("Load FP Model")
    fp_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype = dtype,
        device_map = 'auto'
    )
    fp_model.config.use_cache = False
    logger.info("Load Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast = False
    )
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
    fp_model.eval()
    logger.info(fp_model)

    # Step 2-2: Load Calibration Dataset
    calib_datasets = [x.strip() for x in args.calib_datasets.split(',')]
    calib_num_samples = [int(x.strip()) for x in args.calib_num_samples.split(',')]
    assert len(calib_datasets) == len(calib_num_samples)

    samples, samples_mask, pivot = get_calib_dataset(
        calib_datasets, 
        calib_num_samples, 
        tokenizer, 
        'approx',
        val_ratio = args.calib_val_ratio,
        seed = args.m_seed, 
        logger = logger,
        calib_max_length = args.calib_max_length
    )
    setattr(args, 'calib_val_pivot', pivot)
    logger.info("Calibration datset is loaded")
    logger.info(f"Samples\t{samples.shape}")
    logger.info(f"Samples Mask\t{samples_mask.shape}")

    logger.info(samples)

    # Step 2-3: Extract FP Activation
    tot_iter = len(samples) // args.approx_batch_size
    if len(samples) % args.approx_batch_size != 0:
        tot_iter +=1

    logger.info("Extract FP Activation")
    with torch.no_grad():
        for iter_i in tqdm(range(tot_iter), desc='extract activations'):
            each_sample = samples[iter_i * args.approx_batch_size: (iter_i+1) * args.approx_batch_size].cuda()
            each_mask = samples_mask[iter_i * args.approx_batch_size: (iter_i+1) * args.approx_batch_size].cuda()
            if iter_i != 0:
                if args.a_type == 'hid':
                    pred_out = fp_model.model(
                        input_ids = each_sample,
                        attention_mask = each_mask
                    )[0].cpu()
                elif args.a_type == 'logit':
                    pred_out = fp_model(
                        input_ids = each_sample,
                        attention_mask = each_mask
                    )[0].cpu()
                else: raise Exception(f'{args.a_type} type Activation - Not Implemented')

                outs = torch.cat((
                    outs,
                    pred_out
                ), dim = 0)
                del pred_out
                gc.collect()
                torch.cuda.empty_cache()
            else:
                if args.a_type == 'hid':
                    outs = fp_model.model(
                        input_ids = each_sample,
                        attention_mask = each_mask
                    )[0].cpu()
                elif args.a_type == 'logit':
                    outs = fp_model(
                        input_ids = each_sample,
                        attention_mask = each_mask
                    )[0].cpu()
    del each_sample
    del each_mask
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Extract FP Activation - Done")

    # Step 2-4: Delete FP Model
    del fp_model
    gc.collect()
    torch.cuda.empty_cache()

    #==========================================================================================
    # Step 3: Load Quantized Model
    #==========================================================================================

    # Step 3-1: Load Fake Quantized Model & Attach LoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.q_model,
        torch_dtype = dtype,
        device_map = 'auto',
        trust_remote_code=True 
    )

    target_linear = ['gate_proj', 'k_proj', 'o_proj', 'v_proj', 'q_proj', 'up_proj', 'down_proj']
    target_t_type = 'CAUSAL_LM'

    lora_config = LoraConfig(
        init_lora_weights = "gaussian",
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        target_modules = target_linear,
        lora_dropout = args.lora_dropout,
        bias = "none",
        task_type = target_t_type 
    )
    logger.info("Load zero initialized LoRA Adapter")
    model = get_peft_model(model, lora_config)
    logger.info(f"LoRA Config: {lora_config}")
    model.config.use_cache = False
    logger.info(model)

    # Step 3-2: Print Trainable parameters
    logger.info("\n\n\n\n ========== print trainable parameters ================\n")
    for name, para in model.named_parameters():
        if para.requires_grad==True:
            logger.info(name)
    logger.info("=================================================")


    #==========================================================================================
    # Step 4: RILQ
    #==========================================================================================

    # Step 4-1: Prepare Tuning
    max_device = max(model.hf_device_map.values())
    min_device = min(model.hf_device_map.values())
    logger.info(f"max device: {max_device}")
    logger.info(f"min device: {min_device}")

    if args.a_type =='hid':
        outs = outs.to(max_device)

    target_linears = MODEL_INFO[args.base_model.split('/')[-1]]['linears']
    named_linears = get_target_modules(
        model, 
        targets = target_linears
    )

    loss_fn = nn.MSELoss(reduction = 'mean')
    optimizer = optim.Adam([x for x in model.parameters() if x.requires_grad], lr = args.approx_lr)

    # Training, Validation Input
    t_i_set = samples[args.calib_val_pivot:].to(min_device)
    v_i_set = samples[:args.calib_val_pivot].to(min_device)

    # Training, Validation Mask
    t_m_set = samples_mask[args.calib_val_pivot:].to(min_device)
    v_m_set = samples_mask[:args.calib_val_pivot].to(min_device)

    # Training, Validation Output
    t_o_set = outs[args.calib_val_pivot:]
    v_o_set = outs[:args.calib_val_pivot]

    logger.info(f"Training data: {len(t_o_set)}")
    logger.info(f"Validation data: {len(v_o_set)}")

    nots = len(t_i_set)
    batch_size = args.approx_batch_size
    
    # Step 4-2: Validation

    def validate(model, v_input, v_mask, v_output, loss_fn):
        batch_size = args.approx_batch_size
        val_steps = math.ceil(v_input.shape[0]/args.approx_batch_size)
        val_a_loss = []
        val_gt_loss = []
        val_loss = []
        losses = {}
        with torch.no_grad():
            loss = torch.tensor(0.0, device = max_device)

            for v_step_i in range(val_steps):
                v_batch_input = v_input[v_step_i*batch_size:(v_step_i+1)*batch_size]
                v_batch_mask = v_mask[v_step_i*batch_size:(v_step_i+1)*batch_size]
                v_batch_label = v_input[v_step_i*batch_size:(v_step_i+1)*batch_size].to(max_device)

                v_batch_output = v_output[v_step_i*batch_size:(v_step_i+1)*batch_size]
                if args.a_type =='logit':
                    v_batch_output = v_batch_output.to(min_device)
                
                if args.a_type =='logit':
                    v_b_act_out = model.model(input_ids = v_batch_input, attention_mask = v_batch_mask)[0]
                else:
                    v_b_act_out = model.model.model(input_ids = v_batch_input, attention_mask = v_batch_mask)[0]

                v_act_loss = loss_fn(v_b_act_out, v_batch_output)

                val_a_loss.append(v_act_loss.item())
                
                v_gt_loss = get_gt_loss(model, v_batch_label, v_b_act_out, True if args.a_type =='logit' else False)
                val_gt_loss.append(v_gt_loss.item())

            
            v_gt_loss = sum(val_gt_loss) / len(val_gt_loss)
            losses['v_gt_loss'] = v_gt_loss
            loss = loss + (args.gt_weight * v_gt_loss)
            
            v_act_loss = sum(val_a_loss)/len(val_a_loss)
            losses['v_a_loss'] = v_act_loss
            loss = loss + (args.a_dis_weight * v_act_loss)

        return loss, losses

    # Initial Validation Loss
    model.eval()
    val_loss, val_losses = validate(
        model = model, 
        v_input = v_i_set, 
        v_mask = v_m_set,
        v_output = v_o_set ,
        loss_fn = loss_fn,
    )
    model.train()
    gc.collect()
    torch.cuda.empty_cache()


    # Save Log
    approx_log = []
    base_v_log = {
        'stage': 'approx',
        'step': 0,
        'v_loss': val_loss.item()
    }
    base_v_log.update(val_losses)
    approx_log.append(base_v_log)
    print(approx_log)

    # Step 4-3: RILQ Iteration
    best = float('inf')
    es_count = 0
    tr_steps = math.ceil(nots/args.approx_batch_size)

    for step_i in range(1, args.approx_total_steps+1):
        s_time = time.time()
        on_loading_time = 0.0 

        # Step 4-3-1: Prepare Inputs, Outputs
        step_i_1 = step_i - 1
        cur_batch = step_i_1 % tr_steps
        batch_input = t_i_set[cur_batch*batch_size: (cur_batch+1)*batch_size]
        batch_mask = t_m_set[cur_batch*batch_size: (cur_batch+1)*batch_size]   

        batch_label = t_i_set[cur_batch*batch_size: (cur_batch+1)*batch_size]
        batch_output = t_o_set[cur_batch*batch_size: (cur_batch+1)*batch_size]

        loss = torch.tensor(0.0, device = max_device)
        tr_losses = {}

        # Step 4-3-2: Model-Loss (Activation Loss)
        if args.a_type == 'logit':
            b_act_out = model.model(input_ids = batch_input, attention_mask = batch_mask)[0]
        else:
            b_act_out = model.model.model(input_ids = batch_input, attention_mask = batch_mask)[0]

        batch_output=batch_output.to(b_act_out.device)
        a_loss = loss_fn(b_act_out, batch_output)
        if args.a_type =='logit':
            a_loss = a_loss.to(max_device)
        
        loss = loss + (args.a_dis_weight*a_loss)

        # Step 4-3-3: GT-Loss
        gt_loss = get_gt_loss(model, batch_label, b_act_out, True if args.a_type =='logit' else False)
        # 40169
        if args.a_type == 'logit':
            gt_loss = gt_loss.to(max_device)
        loss = loss + (args.gt_weight * gt_loss)
    
        # Step 4-3-4: Update LoRA with RILQ-Loss
        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        if (step_i) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        step_time = time.time() - s_time

        # Every Validation Step -> Evaluate & Save Best Model
        if (step_i/args.gradient_accumulation_steps) % args.approx_eval_steps == 0:
            # Step 4-3-5: Evaluate 
            model.eval()
            val_loss, val_losses = validate(
                model = model, 
                v_input = v_i_set, 
                v_mask = v_m_set,
                v_output = v_o_set, 
                loss_fn = loss_fn,
            )
            model.train()
            logger.info(f"Step {step_i} - Total Loss - Validation loss: {val_loss}")

            base_v_log = {
                'stage': 'approx',
                'step': step_i,
                't_a_loss': a_loss.item(),
                't_gt_loss': gt_loss.item(),
                't_loss': loss.item(),
                'v_loss': val_loss.item()
            }
            base_v_log.update(val_losses)
            approx_log.append(
                base_v_log
            )

            # Step 4-3-6: Save Best Model
            if val_loss < best :
                es_count = 0
                best = val_loss
                best_weights = {}
                best_log = base_v_log
                logger.info(f"Save Best Model: {step_i}")
                for b_name, b_module in named_linears.items():
                    with torch.no_grad():
                        assert len(b_module.active_adapter) == 1
                        act_adapter = b_module.active_adapter[0]
                        best_weights[f'{b_name}.lora_A'] = b_module.lora_A[act_adapter].weight.clone().detach().cpu()
                        best_weights[f'{b_name}.lora_B'] = b_module.lora_B[act_adapter].weight.clone().detach().cpu()
                logger.info("Save is Done")

            else:
                if (abs(val_loss - best) >= args.approx_es_delta) or val_loss.isnan().item():
                    es_count+=1   

            # End training when loss is converged
            if args.approx_early_stop and es_count >= args.approx_es_patience:
                break    

        else: # Not Validation Step -> report training loss
            approx_log.append({
                'stage': 'approx',
                'step': step_i,
                't_a_loss': a_loss.item(),
                't_gt_loss': gt_loss.item(),
                't_loss': loss.item() 
            })
    logger.info("Training is Done")

    # Step 4-3-7: (Tuning is Done) Load Best Model 
    logger.info("Dump Logs and Load Best Adapters")
    dump_jsonl(
        join(args.output_dir, 'log', f'approx.whole.log'),
        approx_log
    )
    dump_jsonl(
        join(args.output_dir, 'log', f'best.whole.log'),
        [best_log]
    )
    with torch.no_grad():
        for b_name, b_module in named_linears.items():
            assert len(b_module.active_adapter) == 1
            with torch.no_grad():
                act_adapter = b_module.active_adapter[0]
                device = b_module.lora_A[act_adapter].weight.device
                b_module.lora_A[act_adapter].weight.copy_(best_weights[f'{b_name}.lora_A'].to(device))
                b_module.lora_B[act_adapter].weight.copy_(best_weights[f'{b_name}.lora_B'].to(device))
    logger.info("Load Best Adapters - Done")       

    del best_weights
    gc.collect()
    torch.cuda.empty_cache()

    runtime = time.time() - s_time
    logger.info(f"RunTime: {runtime}")
    logger.info("RILQ is Done")

    #===========================================================================================
    # Step 5: Save Optimized Model
    #===========================================================================================

    lora_model_dir = join(args.output_dir, 'approx_init')
    base_model_dir = args.output_dir
    logger.info("Save Adapters")
    model.save_pretrained(lora_model_dir)
    logger.info("Get Base Model")
    base_model = model.get_base_model()
    logger.info("Unwrap Base Model")
    unwrap_model(base_model)

    base_model_state_dict = base_model.state_dict()
    logger.info("Save Tokenizer")
    tokenizer.save_pretrained(base_model_dir)

    dump_args(args)

    # convert safetensor to bin
    tensors = {}
    with safe_open(join(lora_model_dir, "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    torch.save(tensors, os.path.join(lora_model_dir, "adapter_model.bin"))

    # change adapter_config.json
    with open(join(lora_model_dir, "adapter_config.json"), "r") as fp:
        adapter_config = json.load(fp)
        adapter_config['base_model_name_or_path'] = base_model_dir  # This can be a local path or Hub model id
        adapter_config['init_lora_weights'] = True  
        fp.close()
    with open(join(lora_model_dir, "adapter_config.json"), "w") as fp:
        json.dump(adapter_config, fp, indent=2)


    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info('Define a Dummy Model')

    

    dummy_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True, 
        torch_dtype = dtype
    )

    logger.info('Overwrite Q weight')
    dummy_model.load_state_dict(base_model_state_dict, strict = True)
    logger.info('Save Model')
    dummy_model.save_pretrained(base_model_dir)




















