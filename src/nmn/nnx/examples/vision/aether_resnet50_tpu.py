import os
import sys
import argparse
import warnings
from typing import List, Optional

# JAX / Flax / Optax / Sharding
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from flax import nnx
import optax

# Checkpointing & Logging
import orbax.checkpoint as ocp
import wandb

# Data Loading
from torch.utils.data import DataLoader, IterableDataset, Dataset
from datasets import load_dataset

from torchvision import transforms
from PIL import PngImagePlugin
from tqdm import tqdm
import numpy as np
from nmn.nnx.conv import YatConv

# Fix PIL PNG decompression limit
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)
warnings.filterwarnings('ignore', category=UserWarning)

# -----------------------------------------------------------------------------
# 1. Neural Network Blocks (NNX)
# -----------------------------------------------------------------------------

class BasicBlock(nnx.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nnx.Module] = None, dtype=jnp.float32, rngs: nnx.Rngs = None):
        self.conv1 = YatConv(in_channels, out_channels, kernel_size=(3, 3),
                              strides=stride, padding=1, use_bias=False, constant_alpha=True, dtype=dtype, rngs=rngs)
        self.conv2 = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3),
                              strides=1, padding=1, use_bias=False, dtype=dtype, rngs=rngs)
        self.downsample = downsample

    def __call__(self, x, training: bool = True):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        return out


class Bottleneck(nnx.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nnx.Module] = None, dtype=jnp.float32, rngs: nnx.Rngs = None):
        self.conv1 = nnx.Conv(in_channels, out_channels, kernel_size=(1, 1),
                              use_bias=False, dtype=dtype, rngs=rngs)
        self.conv2 = YatConv(out_channels, out_channels, kernel_size=(3, 3),
                              strides=stride, padding=1, use_bias=False, constant_alpha=True, dtype=dtype, rngs=rngs)
        self.conv3 = nnx.Conv(out_channels, out_channels * self.expansion,
                              kernel_size=(1, 1), use_bias=False, dtype=dtype, rngs=rngs)
        self.downsample = downsample

    def __call__(self, x, training: bool = True):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        return out


class DownsampleBlock(nnx.Module):
    def __init__(self, in_channels, out_channels, stride, dtype, rngs):
        self.conv = nnx.Conv(in_channels, out_channels, kernel_size=(1, 1),
                             strides=stride, use_bias=False, dtype=dtype, rngs=rngs)

    def __call__(self, x, training=True):
        x = self.conv(x)
        return x


class ResNet(nnx.Module):
    def __init__(self, block_cls, layers: List[int], num_classes: int = 1000,
                 dtype=jnp.float32, rngs: nnx.Rngs = None):
        self.in_channels = 64
        self.dtype = dtype

        self.conv1 = nnx.Conv(3, 64, kernel_size=(7, 7), strides=2, padding=3,
                              use_bias=False, dtype=dtype, rngs=rngs)

        # ResNet Layers
        self.layer1 = self._make_layer(block_cls, 64, layers[0], dtype=dtype, rngs=rngs)
        self.layer2 = self._make_layer(block_cls, 128, layers[1], stride=2, dtype=dtype, rngs=rngs)
        self.layer3 = self._make_layer(block_cls, 256, layers[2], stride=2, dtype=dtype, rngs=rngs)
        self.layer4 = self._make_layer(block_cls, 512, layers[3], stride=2, dtype=dtype, rngs=rngs)

        self.feature_dim = 512 * block_cls.expansion
        self.head = nnx.Linear(self.feature_dim, num_classes, use_bias=False, dtype=dtype, rngs=rngs)

    def _make_layer(self, block_cls, out_channels, blocks, stride=1, dtype=jnp.float32, rngs=None):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block_cls.expansion:
            downsample = DownsampleBlock(self.in_channels, out_channels * block_cls.expansion, stride, dtype, rngs)

        layers = []
        layers.append(block_cls(self.in_channels, out_channels, stride, downsample, dtype=dtype, rngs=rngs))
        self.in_channels = out_channels * block_cls.expansion
        for _ in range(1, blocks):
            layers.append(block_cls(self.in_channels, out_channels, dtype=dtype, rngs=rngs))
        
        return nnx.List(layers)

    def __call__(self, x, training: bool = True, return_features: bool = False):
        x = x.astype(self.dtype)
        x = self.conv1(x)
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                x = block(x, training=training)

        x = jnp.mean(x, axis=(1, 2))
        
        if return_features:
            return x

        x = self.head(x)
        return x

# -----------------------------------------------------------------------------
# 2. Data Loading
# -----------------------------------------------------------------------------

class ImageNetStreamDataset(IterableDataset):
    def __init__(self, split='train', transform=None):
        self.dataset = load_dataset('mlnomad/imagenet-1k-224', split=split, streaming=True)
        self.transform = transform

    def __iter__(self):
        for sample in self.dataset:
            try:
                image = sample['image']
                label = sample['label']
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                if self.transform:
                    image = self.transform(image)
                yield image, label
            except Exception:
                continue

class ImageNetMapDataset(Dataset):
    def __init__(self, split='train', transform=None, cache_dir=None, keep_in_memory=False):
        self.dataset = load_dataset('mlnomad/imagenet-1k-224', split=split, streaming=False, 
                                    cache_dir=cache_dir, keep_in_memory=keep_in_memory)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            sample = self.dataset[idx]
            image = sample['image']
            label = sample['label']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            raise e

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_numpy_batch(batch):
    images, labels = batch
    # PyTorch NCHW -> JAX NHWC
    images_np = images.numpy().transpose(0, 2, 3, 1)
    labels_np = labels.numpy()
    return images_np, labels_np

# -----------------------------------------------------------------------------
# 3. Training Step (NNX + Mesh)
# -----------------------------------------------------------------------------

@nnx.jit
def train_step(model, optimizer, batch_images, batch_labels):
    
    def loss_fn(model):
        outputs = model(batch_images, training=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(outputs, batch_labels).mean()
        acc = jnp.mean(jnp.argmax(outputs, axis=1) == batch_labels)
        return loss, acc

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(model)
    
    optimizer.update(model, grads)
    
    return loss, aux

@nnx.jit
def val_step(model, batch_images, batch_labels):
    outputs = model(batch_images, training=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(outputs, batch_labels).mean()
    acc = jnp.mean(jnp.argmax(outputs, axis=1) == batch_labels)
    return loss, acc

# -----------------------------------------------------------------------------
# 4. Model Initialization Helper
# -----------------------------------------------------------------------------

@nnx.jit(static_argnames=('block_cls', 'layers', 'num_classes', 'dtype', 'lr', 'epochs'))
def create_model_and_optimizer(block_cls, layers, num_classes, dtype, lr, epochs):
    rngs = nnx.Rngs(0)
    model = ResNet(
        block_cls=block_cls, layers=layers, num_classes=num_classes,
        dtype=dtype, rngs=rngs
    )
    
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=epochs * 1000)
    optimizer = nnx.Optimizer(model, optax.adamw(schedule, weight_decay=0.01), wrt=nnx.Param)
    return model, optimizer

# -----------------------------------------------------------------------------
# 5. Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Model Args
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--batch-size', type=int, default=1024, help="Global batch size across all devices")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--mixed-precision', action='store_true', default=True)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cache-dir', type=str, default=None, help='Directory to cache the dataset')
    parser.add_argument('--in-memory', action='store_true', help='Cache the dataset in RAM (keep_in_memory=True)')
    parser.add_argument('--streaming', action='store_true', help='Use streaming mode (default: False)')
    
    # Checkpointing & Logging Args
    parser.add_argument('--save-dir', type=str, default='./checkpoints_flax')
    parser.add_argument('--checkpoint-keep', type=int, default=3, help='Number of checkpoints to keep')
    parser.add_argument('--wandb-project', type=str, default="imagenet-flax", help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, default="irf-sic", help='WandB entity/username')
    parser.add_argument('--wandb-name', type=str, default=None, help='WandB run name')
    
    parser.add_argument('-f', '--file', type=str, default='', help='Jupyter kernel file (ignored)')

    # Use parse_known_args to avoid crashing on Jupyter/IPython arguments
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown arguments: {unknown}")

    # 1. Initialize WandB
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args)
        )
        print(f"WandB initialized: {args.wandb_project}")

    # 2. Setup Orbax
    ckpt_dir = os.path.abspath(args.save_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Use StandardCheckpointer for PyTrees
    options = ocp.CheckpointManagerOptions(max_to_keep=args.checkpoint_keep, create=True)
    mngr = ocp.CheckpointManager(ckpt_dir, ocp.StandardCheckpointer(), options)
    print(f"Orbax Checkpoint Manager initialized at {ckpt_dir}")

    # -------------------------------------------------------------------------
    # MESH SETUP
    # -------------------------------------------------------------------------
    # Simulate devices if running on CPU-only local machine for testing
    if jax.default_backend() == 'cpu' and len(jax.devices()) == 1:
        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
        if not jax._src.xla_bridge.backends_are_initialized():
             jax.config.update('jax_num_cpu_devices', 8)

    devices = jax.devices()
    num_devices = len(devices)
    print(f"JAX Devices: {num_devices} ({devices[0].platform})")

    mesh_shape = (num_devices, 1)
        
    mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape), ('data', 'model'))
    print(f"Mesh Shape: {mesh_shape}, Axis Names: ('data', 'model')")

    # Logical Axis Rules
    # Map 'batch' -> 'data', others (embed, hidden, etc.) to 'model' if desired
    # For ResNet, we primarily care about batch sharding.
    logical_axis_rules = [
        ('batch', 'data'),
        ('embed', 'model'),
        ('features', 'model'),
        ('channels', 'model'),
    ]

    # Enable Eager Sharding
    nnx.use_eager_sharding(True)

    # -------------------------------------------------------------------------
    # MODEL INIT
    # -------------------------------------------------------------------------
    block_map = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
    }
    block_cls, layers = block_map[args.model]
    layers = tuple(layers) # Make hashable for JIT
    dtype = jnp.bfloat16 if args.mixed_precision else jnp.float32

    # Initialize Model under Mesh & Rules Context
    with mesh, nnx.logical_axis_rules(logical_axis_rules):
        model, optimizer = create_model_and_optimizer(
            block_cls, layers, 1000,
            dtype, args.lr, args.epochs
        )

    state = nnx.state((model, optimizer))
    # jax.debug.visualize_array_sharding(state['model']['conv1']['kernel'].value) 

    print(f"Model {args.model} initialized and sharded.")

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------
    if args.streaming:
        print("Using Streaming Dataset")
        train_dataset = ImageNetStreamDataset(split='train', transform=get_transforms(True))
        val_dataset = ImageNetStreamDataset(split='validation', transform=get_transforms(False))
    else:
        print(f"Using Map-Style Dataset (Cache: {args.cache_dir}, In-Memory: {args.in_memory})")
        train_dataset = ImageNetMapDataset(split='train', transform=get_transforms(True), 
                                           cache_dir=args.cache_dir, keep_in_memory=args.in_memory)
        val_dataset = ImageNetMapDataset(split='validation', transform=get_transforms(False), 
                                         cache_dir=args.cache_dir, keep_in_memory=args.in_memory)
    
    if args.batch_size % num_devices != 0:
        raise ValueError(f"Batch size {args.batch_size} must be divisible by device count {num_devices}")
    
    train_shuffle = not args.streaming
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                              drop_last=True, shuffle=train_shuffle)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                            drop_last=True, shuffle=False)

    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------
    best_acc = 0.0
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_metrics = [] # Accuracy
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            global_step += 1
            imgs_np, lbls_np = get_numpy_batch(batch)
            
            data_sharding = NamedSharding(mesh, P('data', None, None, None)) 
            label_sharding = NamedSharding(mesh, P('data'))
            
            imgs_sharded = jax.device_put(imgs_np, data_sharding)
            lbls_sharded = jax.device_put(lbls_np, label_sharding)
            
            loss, aux = train_step(model, optimizer, imgs_sharded, lbls_sharded)
            
            loss_val = float(loss)
            acc_val = float(aux)
            epoch_losses.append(loss_val)
            epoch_metrics.append(acc_val)
            
            # Iteration Logging
            if args.wandb_project:
                wandb.log({
                    'train/iter_loss': loss_val,
                    'train/iter_acc': acc_val,
                    'trainer/global_step': global_step,
                    'epoch': epoch
                })

            pbar.set_postfix({'metric': float(np.mean(epoch_metrics[-10:]))})

        # Calculate epoch averages
        avg_train_loss = np.mean(epoch_losses)
        avg_train_metric = np.mean(epoch_metrics)

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------
        val_acc = 0.0
        val_loss = 0.0
        
        model.eval()
        total_acc = []
        total_loss = []
        for batch in tqdm(val_loader, desc='Val'):
            imgs_np, lbls_np = get_numpy_batch(batch)
            
            # Re-define shardings inside loop or globally (locally here for safety with dynamic mesh)
            data_sharding = NamedSharding(mesh, P('data', None, None, None)) 
            label_sharding = NamedSharding(mesh, P('data'))
            
            imgs_sharded = jax.device_put(imgs_np, data_sharding)
            lbls_sharded = jax.device_put(lbls_np, label_sharding)
            
            loss, acc = val_step(model, imgs_sharded, lbls_sharded)
            total_acc.append(float(acc))
            total_loss.append(float(loss))
        
        val_acc = np.mean(total_acc) * 100
        val_loss = np.mean(total_loss)
        print(f"Epoch {epoch} Val Acc: {val_acc:.2f}%")
        
        # ---------------------------------------------------------------------
        # LOGGING (WandB)
        # ---------------------------------------------------------------------
        if args.wandb_project:
            log_dict = {
                'epoch': epoch,
                'train/loss': avg_train_loss,
                'train/metric': avg_train_metric,
                'val/loss': val_loss,
                'val/acc': val_acc,
            }
            wandb.log(log_dict)

        # ---------------------------------------------------------------------
        # CHECKPOINTING (Orbax)
        # ---------------------------------------------------------------------
        should_save = False
        if val_acc > best_acc:
            best_acc = val_acc
            should_save = True
            print(f"New best model: {best_acc:.2f}%")

        if should_save:
            # Get current state (parameters + optimizer stats)
            raw_state = nnx.state((model, optimizer))
            
            # Save using Orbax
            save_args = ocp.args.StandardSave(raw_state)
            mngr.save(step=epoch, args=save_args)
            mngr.wait_until_finished() # Ensure save completes before proceeding
            print(f"Checkpoint saved for epoch {epoch}")

    if args.wandb_project:
        wandb.finish()

if __name__ == '__main__':
    main()