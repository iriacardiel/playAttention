import os
import json
import datetime
import torch
import copy
import time
import psutil
import torch.multiprocessing
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from synthetic_pulse_generator.dataset_pytorch import EmitterDataset
from model.model_pytorch import DeinterleavingModel, SaveImgOnEpoch
from model.custom_loss import smcf_loss
from torch.nn import DataParallel

# Adalberto Claudio: Changes introduced in Distributed training, compiling, and gradient accumulation.
from torch.nn.parallel            import DistributedDataParallel as DDP
from torch.utils.data_distributed import DistributedSampler

# TODO: meter jamming
# TODO: probar con varias gpus?
# TODO: tener en cuenta el PW del pulso generado para ver ocultaciones. Como solo se usa el PRI no afecta, aunque la perdida de pulso podria tener un patrón en vez de ser aleatoria.

class Datasets:
    def __init__(self):
        self.train_dataset = None
        self.val_dataset = None
        self.dataloader_train = None
        self.dataloader_val = None
        self.pool_emitters_train = None
        self.pool_emitters_val = None
        self.peak_ram_used = 0

    def sparse_collate_fn(self, batch):
        data = [item[0] for item in batch]  # Build list directly
        labels = []

        for item in batch:
            dense_matrix = np.zeros(item[1][0], dtype=np.int16)
            dense_matrix[tuple(item[1][1])] = item[1][2]
            labels.append(dense_matrix)

        return torch.tensor(np.array(data), dtype=torch.float32), torch.tensor(np.stack(labels), dtype=torch.int8)

    def generate_datasets(self, conf, rank, world_size):
        # Since the generation internally uses multiprocessing, it is not possible to parallelize the generation
        # of the training and validation dataset.
        self.train_dataset, self.pool_emitters_train = EmitterDataset(conf, pool_size=conf["cfg_generation"]["pool_size_train"],
                                                                      save_path=conf["cfg_generation"]["path_emitters_train_csv"],
                                                                      dataset_path=None,
                                                                      pool_emitters=self.pool_emitters_train).generate()

        self.peak_ram_used = max(self.peak_ram_used, psutil.Process().memory_info().rss)

        if conf["cfg_generation"]["new_pool_per_epoch"]:
            del self.pool_emitters_train
            self.pool_emitters_train = None

        self.val_dataset, self.pool_emitters_val = EmitterDataset(conf,
                                                                  pool_size=conf["cfg_generation"]["pool_size_val"],
                                                                  save_path=conf["cfg_generation"]["path_emitters_val_csv"],
                                                                  dataset_path=None,
                                                                  pool_emitters=self.pool_emitters_val).generate()

        self.peak_ram_used = max(self.peak_ram_used, psutil.Process().memory_info().rss)

        if conf["cfg_generation"]["new_pool_per_epoch"]:
            del self.pool_emitters_val
            self.pool_emitters_val = None

        # Dataloader arguments.  
        # Adalberto Claudio: Changes introduced in Distributed training, compiling, and gradient accumulation.
        '''
        [Previous code stated this] Using pin_memory and workers is slower than don't use them in this case.
        This is not correct. Usage varies depending on setting:
            1. pin_memory: pins host RAM so data can be copied faster to the GPU with direct-memory-access.
            2. tensor.to(non_blocking=True): Tensor must reside in pinned memory. Asynchronous copy of data to GPU, call returns as soon as possible instead of waiting for CUDA to finish.
            3. num_workers: number of forked worker processes that load and transform samples in parallel. 

        Settings:
            - Maximum GPU throughput: pin_memory=True, non_blocking=True, num_workers as high as possible so GPU never waits.
            - Debugging / low-RAM: pin_memory=False, non_blocking=False.
            - CPU-only: pin_memory and non_blocking have no effect. 
        '''
        # Distributed samplers.
        self.train_sampler = DistributedSampler(self.train_dataset, rank=rank, num_replicas=world_size) 
        self.eval_sampler  = DistributedSampler(self.val_dataset, rank=rank, num_replicas=world_size) 
        # Dataloaders and arguments.
        train_kwargs  = {'batch_size': conf["cfg_model"]["batch_size"], 'shuffle':True, 'collate_fn': self.sparse_collate_fn, 'pin_memory':True, 'num_workers': 4, 'sampler':self.train_sampler}
        eval_kwargs   = {'batch_size': conf["cfg_model"]["batch_size"], 'shuffle':True, 'collate_fn': self.sparse_collate_fn, 'pin_memory':True, 'num_workers': 4, 'sampler':self.eval_sampler}
        self.dataloader_train = DataLoader(self.train_dataset, **data_kwargs)
        self.dataloader_val  = DataLoader(self.val_dataset, **data_kwargs)

        self.peak_ram_used = max(self.peak_ram_used, psutil.Process().memory_info().rss)

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_dataloader_train(self):
        return self.dataloader_train

    def get_dataloader_val(self):
        return self.dataloader_val

# Adalberto Claudio: Changes introduced in Distributed training, compiling, and gradient accumulation.
'''
    Enable print options only for master process.
'''
def setup_for_distributed(is_master):
    import buitlins as __bultin__
    bultiin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            bultin_print(*args, **kwargs)

    __builtin__.print = print

# Adalberto Claudio: Changes introduced in Distributed training, compiling, and gradient accumulation.
'''
    Setup variables for DDP run.
'''
def setup_distributed():

    # Gather variables for distributed learning.
    rank        = None
    local_rank  = None
    world_size  = torch.cuda.device_count()
    distributed = False
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank  = int(os.environ['LOCAL_RANK'])
        rank        = int(os.environ['RANK'])
        distributed = True

    elif 'SLURM_PROCID' in os.environ:
        world_size              = int(os.environ['WORLD_SIZE'])
        os.eviron['RANK']       = os.environ['SLURM_PROCID']
        rank                    = int(os.environ['RANK'])
        gpus_per_node           = int(os.environ['SLURM_GPUS_ON_NODE']) # Visible GPUs per node, it assumes the number is the same per node.
        local_rank              = rank - gpus_per_node * (rank // gpus_per_node)
        os.eviron['LOCAL_RANK'] = str(local_rank)
        distributed             = True

    # Distributed Learning Initialization. 
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    # Set device for process. 
    torch.cuda.set_device(local_rank)

    # Wait for all processes.
    torch.distributed.barrier()

    # Setup print only for the master process.
    setup_for_distributed(is_master=(rank==0))

    return [rank, local_rank, world_size, distributed, rank==0]

'''
    Train method
        directory
        conf
        generate_datasets_flag
        datasets
        distributed_setup
        ctx
'''
# Adalberto Claudio: Changes introduced in Distributed training, compiling, and gradient accumulation.
def run_train(directory, conf, generate_datasets_flag, datasets, distributed_setup, ctx):

    # Retrieve information about DDP setup.
    rank, local_rank, world_size, master = distributed_setup

    # Get start metrics
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize variables.
    best_loss = float('inf')
    early_stopping_counter = 0

    # Create directories and files.
    today = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    conf["exp_dir"] = directory + '/trains/' + today

    os.makedirs(directory + '/trains/', exist_ok=True)
    checkpoint_path = os.path.expanduser(directory + '/trains/' + today + "/model.ckpt")

    os.mkdir(directory + '/trains/' + today)
    if master:
        with open(conf["exp_dir"] + '/json_data.json', 'w') as outfile:
            json.dump(conf, outfile, indent=4)

    save_img = SaveImgOnEpoch(conf["exp_dir"] + "/history.png", lr=conf["cfg_model"]["learning_rate"],
                              n_heads=conf["cfg_model"]["n_heads"], n_encoders=conf["cfg_model"]["n_encoders"])

    # Generate dataset previously to train the model (if the generation in each epoch is not enabled by user).
    if generate_datasets_flag and not conf['cfg_generation']["new_dataset_per_epoch"]:
        conf["cfg_generation"]["new_pool_per_epoch"] = True
        datasets.generate_datasets(conf, rank, world_size)

    # Create the model.
    start_time_train = time.time()
    model = DeinterleavingModel(d_model=conf["cfg_model"]["model_dim"],
                                input_sequence_size=conf["cfg_model"]["max_seq_len"],
                                n_encoders=conf["cfg_model"]["n_encoders"],
                                n_heads=conf["cfg_model"]["n_heads"],
                                dropout=conf["cfg_model"]["dropout"],
                                input_vars=conf["cfg_model"]['input_vars'],
                                device=local_rank)

    # Compile the model. 
    if conf['cfg_model']['compile']:
        model = torch.compile(model)

    # Parallelize the model into multiple GPUs
    model = DDP(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=conf["cfg_model"]["learning_rate"])

    # Dataloaders and Distributed samplers.
    train_loader = datasets.get_dataloader_train()
    val_loader   = datasets.get_dataloader_val()

    # Train the model
    print('** Training the model **')
    # Grad scaler, used whenever you train on half-precision (float16 or bfloat16). IT prevents small gradients from flushing to zero (multiplying gradients by power-of-two scale factor) 
    # and large ones from overflowing (divides the gradients by the same factor).
    # NOTE 1: Use full precision for debugging enabled=False.
    # NOTE 2: Save and restore scaler.state_dict() with the model and optimizer, it resumes training with the same scale.
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype=='float16'))

    # Set counter for gradient accumulation steps before optimizer´s step. 
    grad_acc_counter = 0

    # Iterate through training.
    for epoch in range(conf['cfg_model']["epochs"]):
        datasets.train_sampler.set_epoch(epoch)

        # If new datasets must be generated in each epoch.
        if conf["cfg_generation"]["new_dataset_per_epoch"]:
            datasets.generate_datasets(conf)
            train_loader = datasets.get_dataloader_train()
            val_loader = datasets.get_dataloader_val()

        model.train()
        train_loss = torch.zeros().to(local_rank)

        # Train set iteration.
        for x_batch, y_batch in tqdm(train_loader):
            
            # Data to GPU.
            x_batch = x_batch.to(local_rank, non_blocking=True) 
            y_batch = y_batch.to(local_rank, non_blocking=True)

            # Take gradient accumulation + 1 and take optimizer step.
            if grad_acc_counter == conf['grad_acc_steps'] - 1:
                
                # Forward pass.
                with ctx:
                    outputs = model(x_batch)
                    loss = smcf_loss(local_rank, outputs, y_batch)

                # Calculate gradients.
                scaler.scale(loss).backward()

                # Take gradient optimizer step.
                scaler.step(optimizer)
                scaler.update()
            
                # Set gradients to zero.
                optimizer.zero_grad(set_to_none=True)

                # Gradient accumulation counter to zero.
                grad_acc_counter = 0 

            else:
                # Gradient accumulation, no sync across GPUs.
                with model.no_sync():

                    # Forward pass.
                    with ctx:
                        outputs = model(x_batch)
                        loss = smcf_loss(local_rank, outputs, y_batch)

                    # Calculate gradients.
                    scaler.scale(loss).backward()

                    # Increase gradient accumulater counter.
                    grad_acc_counter += 1

            train_loss += loss
        # Average for the entire loader, this a running average. We trade accuracy of train loss for execution time.
        train_loss /= len(train_loader)

        # Wait for all GPUs to start evaluation.
        dist.barrier()  

        # Gather across GPUs.  
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)   
        train_loss.item()

        # Set model to evaluation mode: no gradient and rolling history for some parameters.
        model.eval()      
        
        # Iterate through validation set. 
        val_loss = torch.zeros().to(local_rank)
        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader):
                # Data to GPU.
                x_batch = x_batch.to(local_rank, non_blocking=True) 
                y_batch = y_batch.to(local_rank, non_blocking=True)
                # Forward pass.
                outputs = model(x_batch)
                loss = smcf_loss(local_rank, outputs, y_batch)
                val_loss += loss
        val_loss /= len(val_loader)
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss.item()

        # Wait for all GPUs to finish.
        dist.barrier()  

        print(f"\nEpoch {epoch}/{conf['cfg_model']['epochs']}, "f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")

        if val_loss < best_loss:
            best_loss = val_loss
            time.sleep(10)
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= conf['cfg_model']["early_stopping_patience"]:
            print("Early stopping triggered.")
            break

        save_img.on_epoch_end(train_loss, val_loss)

    # Adalberto Claudio: Changes introduced in Distributed training, compiling, and gradient accumulation.
    # Reconsider moving to Weights & Biases, it keeps track of this data. 
    # Get final metrics
    ram_info = psutil.virtual_memory()
    peak_ram_used = max(datasets.peak_ram_used, psutil.Process().memory_info().rss)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_train_time = end_time - start_time_train

    # Save performance to file
    if master:
        with open(conf["exp_dir"] + "/resources_log.txt", "w") as f:
            f.write(f"Execution Time: {execution_time:.2f} seconds\n")
            f.write(f"Generate Dataset: {generate_datasets_flag}\n")
            f.write(f"Execution Train Time: {execution_train_time:.2f} seconds\n")
            f.write(f"Epochs: {epoch}\n")
            f.write(f"Best Validation Loss: {best_loss}\n")
            f.write(f"Total RAM: {ram_info.total / (1024 ** 3):.2f} GB\n")
            f.write(f"Peak RAM Used: {peak_ram_used / (1024 ** 3):.2f} GB\n")
            f.write(f"Percentage RAM Used: {100.0 * peak_ram_used / ram_info.total }%\n")

            if torch.cuda.is_available():
                f.write(f"GPU Device: {local_rank}\n")
                f.write(f"Current Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB\n")
                f.write(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB\n")
                f.write(f"Current Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB\n")
                f.write(f"Max Memory Reserved: {torch.cuda.max_memory_reserved() / (1024 ** 2):.2f} MB\n")

        else:
            f.write(f"No GPU available.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=local_rank)
    
    # Save model in ONNX format
    if conf["cfg_model"]["save_model_onnx"] and master:
        path_model = directory + "trains/" + today + "/" + "model.onnx"
        torch_input = torch.ones(1, conf["cfg_model"]["max_seq_len"], 1)
        torch.onnx.export(model, torch_input, path_model, input_names=['input'], output_names=['output'])


if __name__ == "__main__":
        

    # Adalberto Claudio: Changes introduced in Distributed training, compiling, and gradient accumulation.
    # Distributed learning setup. 
    rank, local_rank, world_size, distributed, master = setup_distributed()

    # System.
    device = 'cuda'     # E.g. 'cuda', 'cpu', or 'mps' on mackbooks.
    dtype  = 'bfloat16' # 'float32', 'bfloat32', or 'float16'.

    # Mixed precision handling. 
    torch.backends.cuda.matmul.allow_tf32 = True   # Allows tf32 on matmul.
    torch.backends.cudnn.allow_tf32       = True   # Allows tf32 on cudnn.
    device_type                           = {'float32'; torch.float32, 'float16'; torch.float16, 'bfloat16'; torch.bfloat16}[dtype]             # Data type assignation dictionary. 
    ctx                                   = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=dtype) # Automatic mixed precision handling (GPU) or null context (CPU).

    # Define device to run; Old code. 
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #count = torch.cuda.device_count()
    #print(f"Using device: {device}/ Number of GPUs {count}")

    # Auxiliar copy of the configuration to check if any parameter has changed between experiments.
    conf_copy = {}

    # Initialize training and validation datasets.
    datasets = Datasets()

    # Path in which experiments are defined and the results will be stored.
    experiments_dir = './data/experiments'
    if master:
        os.makedirs(experiments_dir, exist_ok=True)
    # For each experiment defined.
    for item in sorted(os.listdir(experiments_dir)):

        exp_dir = os.path.join(experiments_dir, item)
        print(f"Experiment directory: {str(exp_dir)}")

        #Check if the experiment has been done previously. Otherwise, run the experiment,
        if os.path.exists(exp_dir + '/trains/'):
            print(f"Experiment has been trained before. To run again remove trains directory")
            continue

        # Load configurations of the experiment.
        cfg_model = []
        if master:
            with open(exp_dir + '/config/config_model.json') as f:
                cfg_model.append(json.load(f))
        cfg_model = cfg_model[0]

        cfg_generation = []
        if master:
            with open(exp_dir + '/config/config_generation.json') as f:
                cfg_generation.append(json.load(f))
        cfg_generation = cfg_generation[0]
        cfg_generation["train_mode"] = True

        cfg_emitters = []
        if master:
            with open(exp_dir + '/config/config_emitters.json') as f:
                cfg_emitters.append(json.load(f))
        cfg_emitters = cfg_emitters[0]

        conf = {'cfg_model': cfg_model, 'cfg_generation': cfg_generation, 'cfg_emitters': cfg_emitters}

        # Check if any parameter has change between the current and previous experiments
        # in order to generate the dataset again.
        generate_datasets_flag = True
        if len(conf_copy) != 0 and \
                conf['cfg_model']['max_seq_len'] == conf_copy['cfg_model']['max_seq_len'] and \
                conf['cfg_generation'] == conf_copy['cfg_generation'] and \
                conf['cfg_emitters'] == conf_copy['cfg_emitters']:
            generate_datasets_flag = False

        # Copy the current experiment configuration to check in the next experiment.
        conf_copy = copy.deepcopy(conf)

        # Training
        conf['grad_acc_steps'] = 4
        run_train(exp_dir, conf, generate_datasets_flag, datasets, [rank, local_rank, world_size, master], ctx)

