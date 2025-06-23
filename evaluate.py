import re
import sys
import io, os
import torch
import numpy as np
import logging
import tqdm
import fcntl
import time
import argparse
from prettytable import PrettyTable
import transformers
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from senllm import LlamaForCausalLM, Qwen2ForCausalLM, Gemma2ForCausalLM
from colorama import Fore, Style
import textwrap
from scipy.stats import spearmanr
import numpy as np
import yaml
import warnings
warnings.filterwarnings("ignore")


if torch.cuda.is_available():
    print("We are using GPU!")
    torch.cuda.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)

COEFF = float(os.environ.get("COEFF", 1.0))

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def lock_and_write_file(file_path, content):
    with open(file_path, 'a') as file:
        while True:
            try:
                # Acquire an exclusive lock (non-blocking)
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Perform your write operations here
                file.write(content + '\n')
                file.flush()

            except IOError as e:
                print("File is locked by another process. Can't write.")
                time.sleep(1)
            finally:
                # Release the lock
                fcntl.flock(file, fcntl.LOCK_UN)
                break

def load_config_from_yaml(config_file="config.yaml", config_name=None):

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"warning: config file {config_file} not found, using command line parameters")
        return None
    except yaml.YAMLError as e:
        print(f"error: config file {config_file} format error: {e}")
        return None
    
    if config_name is None:
        config_name = yaml_config.get('default_config', 'llama-2-7b')
    
    if config_name not in yaml_config.get('models', {}):
        available_configs = list(yaml_config.get('models', {}).keys())
        print(f"error: config '{config_name}' not found")
        print(f"available configs: {available_configs}")
        return None
    
    # 获取指定配置
    config = yaml_config['models'][config_name].copy()
    
    # 添加GPU配置
    if 'gpu_config' in yaml_config:
        config['gpu_config'] = yaml_config['gpu_config']
    
    print(f"✓ successfully loaded config: {config_name}")
    return config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None,
                        help="config name, read parameters from config.yaml")
    parser.add_argument("--config_file", type=str, default="config.yaml",
                        help="config file path")
    

    parser.add_argument("--tokenizer_name", type=str, 
                        default='')
    parser.add_argument("--model_name_or_path", type=str,
                        help="Transformers' model name or path")
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na', 'stsb'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument('--tensor_parallel', action='store_true')
    parser.add_argument('--prompt_method', type=str, 
                        default='prompteol', choices=['prompteol', 'metaeol', 'cot', 'ke'], help="What prompt method to use.")
    parser.add_argument("--use_which_plan", type=str,
                        choices=['tp', 'vanilla'],
                        default='tp')
    parser.add_argument("--output_layer", type=int, 
                        default=-1)
    parser.add_argument("--tp_starting_index", type=int, 
                        default=1)
    parser.add_argument("--tp_exiting_index", type=int, 
                        default=99)
    parser.add_argument("--batch_size", type=int, 
                        default=16)

    args = parser.parse_args()
    
    if args.config:
        config = load_config_from_yaml(args.config_file, args.config)
        if config is None:
            print("config loading failed, exit program")
            sys.exit(1)
        
        args.model_name_or_path = config.get('model_name_or_path', args.model_name_or_path)
        args.use_which_plan = config.get('use_which_plan', args.use_which_plan)
        args.output_layer = config.get('output_layer', args.output_layer)
        args.tp_starting_index = config.get('tp_starting_index', args.tp_starting_index)
        args.tp_exiting_index = config.get('tp_exiting_index', args.tp_exiting_index)
        args.batch_size = config.get('batch_size', args.batch_size)
        args.mode = config.get('mode', args.mode)
        args.task_set = config.get('task_set', args.task_set)
        args.prompt_method = config.get('prompt_method', args.prompt_method)
        
        if 'gpu_config' in config and 'cuda_visible_devices' in config['gpu_config']:
            os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_config']['cuda_visible_devices']
            print(f"✓ set GPU devices: {config['gpu_config']['cuda_visible_devices']}")
    
    if not args.model_name_or_path:
        print("error: model path not specified, please use --model_name_or_path parameter or specify it in the config file")
        sys.exit(1)
    hyper_parameters = textwrap.dedent(f"""
        {Fore.CYAN}Configuration:{Style.RESET_ALL}
        {Fore.YELLOW}-------------{Style.RESET_ALL}
        {Fore.GREEN}Backbone                :{Style.RESET_ALL} {args.model_name_or_path.split('/')[-1]}
        {Fore.GREEN}Prompt Method           :{Style.RESET_ALL} {args.prompt_method}
        {Fore.GREEN}Output Layer Index      :{Style.RESET_ALL} {args.output_layer}
        {Fore.GREEN}Plan                    :{Style.RESET_ALL} {args.use_which_plan}
        {Fore.GREEN}TP Starting layer Index :{Style.RESET_ALL} {args.tp_starting_index}
        {Fore.GREEN}TP Exiting layer Index  :{Style.RESET_ALL} {args.tp_exiting_index}
        {Fore.GREEN}Batch Size              :{Style.RESET_ALL} {args.batch_size}
    """)

    print(hyper_parameters)

    if args.tensor_parallel:
        import tensor_parallel as tp
        n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     low_cpu_mem_usage = True, torch_dtype=torch.float16)
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
    else:
        if 'llama' in args.model_name_or_path.lower():
            model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,
                                                        device_map='auto',
                                                        output_hidden_states=True,
                                                        trust_remote_code=True)
            model.model.plan = args.use_which_plan
            model.model.tp_starting_index = args.tp_starting_index
            model.model.tp_exiting_index = args.tp_exiting_index
        elif 'qwen2' in args.model_name_or_path.lower():
            model = Qwen2ForCausalLM.from_pretrained(args.model_name_or_path,
                                                        device_map='auto',
                                                        output_hidden_states=True,
                                                        trust_remote_code=True)
            model.model.plan = args.use_which_plan
            model.model.tp_starting_index = args.tp_starting_index
            model.model.tp_exiting_index = args.tp_exiting_index
        elif 'gemma' in args.model_name_or_path.lower():
            model = Gemma2ForCausalLM.from_pretrained(args.model_name_or_path,
                                                        device_map='auto',
                                                        output_hidden_states=True,
                                                        trust_remote_code=True)
            model.model.plan = args.use_which_plan
            model.model.tp_starting_index = args.tp_starting_index
            model.model.tp_exiting_index = args.tp_exiting_index
        else:
            raise ValueError(f"Cannot find such {args.model_name_or_path.lower()} model!")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = 0  # Set the padding token. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    if args.use_which_plan == 'tp':
        placeholder_token = '<PST>'
        tokenizer.add_tokens([placeholder_token])
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
        
        model.resize_token_embeddings(len(tokenizer))

        embedding_layer = model.get_input_embeddings()
        embedding_layer.weight.requires_grad_(False)
        
        num_dim = embedding_layer.weight.shape[1]
        device = embedding_layer.weight.device
        
        with torch.no_grad():
            embedding_layer.weight[placeholder_token_id] = torch.randn(num_dim, device=device)
        embedding_layer.weight.requires_grad_(True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        if args.mode == 'dev':
            args.tasks = ['STSBenchmark-dev']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'stsb':
        args.tasks = ['STSBenchmark']
    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 32}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 32,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size':args.batch_size}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    if args.prompt_method == "metaeol":
        if args.use_which_plan == 'tp':
            task_prompts = ["In this task, you're presented with a text excerpt. Your task is to categorize the excerpt into a broad category such as 'Education', 'Technology', 'Health', 'Business', 'Environment', 'Politics', or 'Culture'. These categories help in organizing content for better accessibility and targeting. For this task, this sentence : <PST> \"*sent 0*\" should be classified under one general category in one word:\"",
                            "In this task, you're given a statement and you need to determine whether it's presenting an 'Opinion' or a 'Fact'. This distinction is vital for information verification, educational purposes, and content analysis. For this task, this sentence : <PST> \"*sent 0*\" discriminates between opinion and fact in one word:\"",
                            "In this task, you're given a review from an online platform. Your task is to generate a rating for the product based on the review on a scale of 1-5, where 1 means 'extremely negative' and 5 means 'extremely positive'. For this task, this sentence : <PST> \"*sent 0*\" reflects the sentiment in one word:\"",
                            "In this task, you're reading a personal diary entry. Your task is to identify the predominant emotion expressed, such as joy, sadness, anger, fear, or love. For this task, this sentence : <PST> \"*sent 0*\" conveys the emotion in one word:\"",
                            "In this task, you're presented with two sentences. Your task is to assess whether the sentences convey the same meaning. Use 'identical', 'similar', 'different', or 'unrelated' to describe the relationship. To enhance the performance of this task, this sentence : <PST> \"*sent 0*\" means in one word:\"",
                            "In this task, you're given a sentence and a phrase. Your task is to determine if the phrase can be a contextual synonym within the given sentence. Options include 'yes', 'no', or 'partially'. To enhance the performance of this task, this sentence : <PST> \"*sent 0*\" means in one word:\"",
                            "In this task, you're examining a news article. Your task is to extract the most critical fact from the article. For this task, this sentence : <PST> \"*sent 0*\" encapsulates the key fact in one word:\"",
                            "In this task, you're reviewing a scientific abstract. Your task is to identify the main entities (e.g., proteins, diseases) and their relations (e.g., causes, treats). For this task, this sentence : <PST> \"*sent 0*\" highlights the primary entity or relation in one word:\"",
                            ]
        else:
            task_prompts = ["In this task, you're presented with a text excerpt. Your task is to categorize the excerpt into a broad category such as 'Education', 'Technology', 'Health', 'Business', 'Environment', 'Politics', or 'Culture'. These categories help in organizing content for better accessibility and targeting. For this task, this sentence : \"*sent 0*\" should be classified under one general category in one word:\"",
                            "In this task, you're given a statement and you need to determine whether it's presenting an 'Opinion' or a 'Fact'. This distinction is vital for information verification, educational purposes, and content analysis. For this task, this sentence : \"*sent 0*\" discriminates between opinion and fact in one word:\"",
                            "In this task, you're given a review from an online platform. Your task is to generate a rating for the product based on the review on a scale of 1-5, where 1 means 'extremely negative' and 5 means 'extremely positive'. For this task, this sentence : \"*sent 0*\" reflects the sentiment in one word:\"",
                            "In this task, you're reading a personal diary entry. Your task is to identify the predominant emotion expressed, such as joy, sadness, anger, fear, or love. For this task, this sentence : \"*sent 0*\" conveys the emotion in one word:\"",
                            "In this task, you're presented with two sentences. Your task is to assess whether the sentences convey the same meaning. Use 'identical', 'similar', 'different', or 'unrelated' to describe the relationship. To enhance the performance of this task, this sentence : \"*sent 0*\" means in one word:\"",
                            "In this task, you're given a sentence and a phrase. Your task is to determine if the phrase can be a contextual synonym within the given sentence. Options include 'yes', 'no', or 'partially'. To enhance the performance of this task, this sentence : \"*sent 0*\" means in one word:\"",
                            "In this task, you're examining a news article. Your task is to extract the most critical fact from the article. For this task, this sentence : \"*sent 0*\" encapsulates the key fact in one word:\"",
                            "In this task, you're reviewing a scientific abstract. Your task is to identify the main entities (e.g., proteins, diseases) and their relations (e.g., causes, treats). For this task, this sentence : \"*sent 0*\" highlights the primary entity or relation in one word:\"",
                            ]
    elif args.prompt_method == "prompteol":
        if args.use_which_plan == 'tp':
            task_prompts = ['This sentence : <PST> \"*sent 0*\" means in one word:\"']
        else:
            task_prompts = ["This sentence : \"*sent 0*\" means in one word:\""]
    elif args.prompt_method == "cot":
        if args.use_which_plan == 'tp':
            task_prompts = ['After thinking step by step , this sentence : <PST> \"*sent 0*\" means in one word:\"']
        else:
            task_prompts = ['After thinking step by step , this sentence : \"*sent 0*\" means in one word:\"']
    elif args.prompt_method == "ke":
        if args.use_which_plan == 'tp':
            task_prompts = ['The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind , this sentence : <PST> \"*sent 0*\" means in one word:\"']
        else:    
            task_prompts = ['The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind , this sentence : \"*sent 0*\" means in one word:\"']

    print(task_prompts)

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        input_sentences = [' '.join(s) for s in batch]
        if max_length == 500:
            sentences = [tokenizer.decode(tokenizer.encode(s, add_special_tokens=False)[:max_length]) for s in sentences]
            max_length = 512

        new_sentences = []
        for i, s in enumerate(sentences):
            if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
            s = s.replace('"', '\'')
            if len(s) > 0 and '?' == s[-1]: s = s[:-1] + '.'
            for prompt in task_prompts:
                new_sentences.append(prompt.replace('*sent 0*', s).strip())
        sentences = new_sentences

        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
            max_length=max_length,
            truncation=max_length is not None
        )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device) if batch[k] is not None else None
        # Get raw embeddings
        with torch.no_grad():
            raw_outputs = model(output_hidden_states=True, return_dict=True, **batch)
            hidden_states = raw_outputs.hidden_states
            outputs = hidden_states[args.output_layer][:, -1, :]
            outputs = outputs.view(-1, len(task_prompts), outputs.size()[1]).mean(dim=1) # Average the embeddings from different tasks 

            if outputs.dtype == torch.bfloat16:
                # bfloat16 not support for .numpy()
                outputs = outputs.float()


            return outputs.cpu()

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark-dev']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
        #
        # write results and template to file
        if args.task_set != 'transfer':
            with open('./sts-enhance-results', 'a') as f:
                model_name = args.model_name_or_path.split('/')[-1]
                f.write(model_name + ' ' + str(COEFF) + ' ' + str(args.tp_starting_index) + ' ' + ' '.join([str(s) for s in scores]) + '\n')

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

if __name__ == "__main__":
    main()
