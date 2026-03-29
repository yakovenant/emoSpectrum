import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import lr_scheduler, SGD, Adam, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score
from sklearn.manifold import TSNE
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
from losses import AdaptiveCosineCenterCrossEntropyLoss, CosineCenterCrossEntropyLoss, CosCrossEntropyLoss
from nnets import make_model
from dataproc import EmotionDataset, get_stratified_data_splits, get_dataloader, get_dataframe
from visualizer import plot_curve, plot_loss_curves, plot_embeddings_with_dataloader, plot_data_hist
from utils import custom_print, write_print_log

RANDOM_SEED = 678
GPU_ID = 0
N_WORKERS = 0

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


@dataclass
class Hparams:
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    dataset_name: str = "AbstractTTS/iemocap" # iemocap, emotiontalk, dusha
    feature_extractor_name: str = "backbones/facebook/wav2vec2-base"
    model_name: str = "backbones/facebook/wav2vec2-base" # "baselines/wavlm-base", "baselines/hubert-base", "baselines/wav2vec2-base"
    adapter: str = "hybrid_probing" # linear, mlp, hybrid, linear_probing, hybrid_probing
    projector_out_dim: int = 128 # None, 128
    fusion_method: str = "gff" # None=Last, mean=Avg, tws, gff
    stat_pooling: bool = True # True False
    topk_layers: str = False # True False
    num_classes: int = 4
    w_classes: bool = False # False, True
    balanced_data: bool = True
    loss_fn: str = "adaptive_cosine_ce" # "cross_entropy", "contrastive", "cosine_ce", "adaptive_cosine_ce"
    lr_scheduler: str = "reduce_plateau" # "step", "exp", "cos_annealing", "cos_annealing_warmup", "const", "reduce_plateau"
    learning_rate: float = 4e-4
    optimizer_fn: str = "adamw" # "sgd", "adamw"
    n_tolerance: int = 5 # 7 5
    batch_size: int = 128 # 32 56 !!!
    num_epochs: int = 99 # !!!
    sample_rate: int = 16000
    freezing_encoder: bool = True # False True
    gradual_unfreezing: bool = False # False True
    lora: bool = False # True
    augment: bool = True # False True
    dataroot: str = "/media/ssd/datasets/"
    noise_path: str = None
    rir_path: str = None
    n_workers: int = N_WORKERS
    audio_dir = None
    csv_path = None


class ModelTrainer:

    def __init__(self, model, **kwargs):
        self.params = kwargs['args']
        if model.params.projector_out_dim is not None:
            self.params.feat_dim = model.params.projector_out_dim
        else:
            self.params.feat_dim = model.backbone_hidden_size
            if self.params.stat_pooling: self.params.feat_dim *= 2
        assert model.classifier[-1].in_features == self.params.feat_dim
        self.loss = self._init_loss()

        if self.params.adapter == "hybrid_probing":
            if self.params.fusion_method is not None and self.params.fusion_method != "mean":
                if self.params.fusion_method == "gff":
                    fusion_params = list(model.feature_fusion.parameters())
                elif self.params.fusion_method == "tws":
                    fusion_params = [model.feature_fusion]

                if self.params.projector_out_dim is not None:
                    self.optimizer = self._init_optimizer([
                        model.backbone.base_model.encoder.parameters(),
                        fusion_params,
                        model.projector.parameters(),
                        model.classifier.parameters()])
                else:
                    self.optimizer = self._init_optimizer([
                        model.backbone.base_model.encoder.parameters(),
                        fusion_params,
                        model.classifier.parameters()])
            else:
                if self.params.projector_out_dim is not None:
                    self.optimizer = self._init_optimizer([
                        model.backbone.base_model.encoder.parameters(),
                        model.projector.parameters(),
                        model.classifier.parameters()])
                else:
                    self.optimizer = self._init_optimizer([
                        model.backbone.base_model.encoder.parameters(),
                        model.classifier.parameters()])
        elif self.params.adapter == "linear_probing":
            self.optimizer = self._init_optimizer([
                model.backbone.base_model.encoder.parameters(),
                [model.feature_fusion],
                model.classifier.parameters()])
        else:
            raise Exception("Adapter type error!")

        self.scheduler = self._create_scheduler()

    def _init_loss(self):

        if self.params.loss_fn == "cross_entropy":
            if self.params.w_classes:
                criterion = nn.CrossEntropyLoss(weight=self.params.class_weights)
            else:
                criterion = nn.CrossEntropyLoss()
        elif self.params.loss_fn == "cosine_ce":
            if self.params.w_classes:
                criterion = CosineCenterCrossEntropyLoss(ce_weights=self.params.class_weights)
            else:
                criterion = CosineCenterCrossEntropyLoss() # CosCrossEntropyLoss()
        elif self.params.loss_fn == "adaptive_cosine_ce":
            criterion = AdaptiveCosineCenterCrossEntropyLoss(
                num_classes=self.params.num_classes,
                feat_dim=self.params.feat_dim,
                ce_weights=self.params.class_weights,
            )
        elif self.params.loss_fn == "contrastive":
            raise Exception("Not implemented yet.") #criterion = ContrastiveLoss() # TODO
        else:
            raise Exception("Chosen loss is not implemented.")

        criterion.to(self.params.device)
        custom_print(f"\nInitialized loss: {self.params.loss_fn}")

        return criterion

    def _init_optimizer(self, optimizer_parameters=None):

        if optimizer_parameters is None:
            optimizer_parameters = [p for p in self.backbone.parameters() if p.requires_grad]
        elif isinstance(optimizer_parameters, list):
            if len(optimizer_parameters) == 2:
                lr_encoder = self.params.learning_rate * 0.1
                lr_clf = self.params.learning_rate
                if self.params.freezing_encoder:
                    custom_print(f"\nInit classifier LR: {lr_clf}.\n")
                else:
                    custom_print(f"\nInit encoder LR: {lr_encoder},\nInit classifier LR: {lr_clf}.\n")
                if self.params.optimizer_fn == "adam":
                    optimizer = Adam([
                        {
                            'params': optimizer_parameters[0],
                            'lr': lr_encoder
                        },
                        {
                            'params': optimizer_parameters[-1],
                            'lr': lr_clf
                        }
                    ])
                elif self.params.optimizer_fn == "adamw":
                    optimizer = AdamW([
                        {
                            'params': optimizer_parameters[0], # encoder
                            'lr': lr_encoder,
                            'weight_decay': 1e-5
                        },
                        {
                            'params': optimizer_parameters[-1], # classifier
                            'lr': lr_clf,
                            'weight_decay': 1e-5
                        }
                    ])
                elif self.params.optimizer_fn == "sgd":
                    optimizer = SGD([
                        {
                            'params': optimizer_parameters[0],
                            'lr': lr_encoder,
                            'momentum': 0.9,
                            'weight_decay': 1e-2
                        },
                        {
                            'params': optimizer_parameters[-1],
                            'lr': lr_clf,
                            'momentum': 0.95,
                            'weight_decay': 0.0
                        }
                    ])
                else:
                    raise Exception("Chosen optimizer is not implemented.")

            elif len(optimizer_parameters) == 3:
                lr_encoder = self.params.learning_rate * 1.2 # 0.5
                lr_clf = self.params.learning_rate
                if self.params.projector_out_dim is not None:
                    lr_intermediate = self.params.learning_rate * 3 # LR Projector
                    if self.params.freezing_encoder:
                        custom_print(f"\nInit projector LR: {lr_intermediate},\nInit classifier LR: {lr_clf}.\n")
                    else:
                        custom_print(f"\nInit encoder LR: {lr_encoder},\nInit projector LR: {lr_intermediate},\nInit classifier LR: {lr_clf}.\n")
                else:
                    lr_intermediate = self.params.learning_rate * 3 # LR Fusion
                    if self.params.freezing_encoder:
                        custom_print(f"\nInit fusion LR: {lr_intermediate},\nInit classifier LR: {lr_clf}.\n")
                    else:
                        custom_print(f"\nInit encoder LR: {lr_encoder},\nInit fusion LR: {lr_intermediate},\nInit classifier LR: {lr_clf}.\n")

                if self.params.optimizer_fn == "sgd":
                    optimizer = SGD([
                        {
                            'params': optimizer_parameters[0],
                            'lr': lr_encoder,
                            'momentum': 0.9,
                            'weight_decay': 0.0005 # 0.001
                        },
                        {
                            'params': optimizer_parameters[1],
                            'lr': lr_projector,
                            'momentum': 0.95,
                            'weight_decay': 0.0001 # 0.01
                        },
                        {
                            'params': optimizer_parameters[-1],
                            'lr': lr_clf,
                            'momentum': 0.95,
                            'weight_decay': 0.0005 # 0.01
                        }
                    ])
                elif self.params.optimizer_fn == "adamw":
                    optimizer = AdamW([
                        {
                            'params': optimizer_parameters[0], # encoder
                            'lr': lr_encoder,
                            'weight_decay': 1e-5
                        },
                        {
                            'params': optimizer_parameters[1], # projector or fusion
                            'lr': lr_intermediate,
                            'weight_decay': 1e-4
                        },
                        {
                            'params': optimizer_parameters[-1], # classifier
                            'lr': lr_clf,
                            'weight_decay': 1e-5
                        }
                    ])
                else:
                    raise Exception("Chosen optimizer is not implemented.")

            elif len(optimizer_parameters) == 4:
                lr_encoder = self.params.learning_rate * 0.5
                lr_fusion = self.params.learning_rate * 2 # 100
                lr_projector = self.params.learning_rate
                lr_clf = self.params.learning_rate
                custom_print(f"\nInit encoder LR: {lr_encoder},\nInit fusion LR: {lr_fusion},\nInit projector LR: {lr_projector},\nInit classifier LR: {lr_clf}.\n")

                if self.params.optimizer_fn == "sgd":
                    optimizer = SGD([
                        {
                            'params': optimizer_parameters[0],
                            'lr': lr_encoder,
                            'momentum': 0.9,
                            'weight_decay': 0.0005 # 0.001
                        },
                        {
                            'params': optimizer_parameters[1],
                            'lr': lr_fusion,
                            'momentum': 0.9,
                            'weight_decay': 0.0
                        },
                        {
                            'params': optimizer_parameters[2],
                            'lr': lr_projector,
                            'momentum': 0.95,
                            'weight_decay': 0.0001 # 0.01
                        },
                        {
                            'params': optimizer_parameters[-1],
                            'lr': lr_clf,
                            'momentum': 0.95,
                            'weight_decay': 0.0005 # 0.01
                        }
                    ])
                elif self.params.optimizer_fn == "adamw":
                    optimizer = AdamW([
                        {
                            'params': optimizer_parameters[0], # encoder
                            'lr': lr_encoder,
                            'weight_decay': 1e-5
                        },
                        {
                            'params': optimizer_parameters[1], # fusion
                            'lr': lr_fusion,
                            'weight_decay': 1e-4 # 0.0 # !!!
                        },
                        {
                            'params': optimizer_parameters[2], # projector
                            'lr': lr_projector,
                            'weight_decay': 1e-4
                        },
                        {
                            'params': optimizer_parameters[-1], # classifier
                            'lr': lr_clf,
                            'weight_decay': 1e-5
                        }
                    ])
            else:
                raise Exception("Invalid number of parameter groups.")
        else:
            if self.params.optimizer_fn == "adam":
                optimizer = Adam(optimizer_parameters, lr=self.params.learning_rate)
            elif self.params.optimizer_fn == "sgd":
                optimizer = SGD(optimizer_parameters, lr=self.params.learning_rate, momentum=0.9)
            else:
                raise Exception("Chosen optimizer is not implemented.")
        custom_print(f"Initialized optimizer: {self.params.optimizer_fn}\n")

        return optimizer

    def _create_scheduler(self):

        if self.params.lr_scheduler == "step":
            scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        elif self.params.lr_scheduler == "exp":
            scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95, last_epoch=-1)
        elif self.params.lr_scheduler == "const":
            scheduler = lr_scheduler.ConstantLR(self.optimizer, factor=(self.params.learning_rate), total_iters=(self.params.num_epochs//3))
        elif self.params.lr_scheduler == "cos_annealing":
            scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.params.num_epochs)
        elif self.params.lr_scheduler == "cos_annealing_warmup":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2, eta_min=1e-8)
        elif self.params.lr_scheduler == "reduce_plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=self.params.n_tolerance,
                factor=0.5,
                min_lr=1e-6,
                verbose=True
            )
        else:
            raise Exception("Chosen LR scheduler is not implemented.")
        custom_print(f"Initialized LR scheduler: {self.params.lr_scheduler}\n")

        return scheduler

    def training_step(self, model, x, y, get_embeddings=False):

        x, y = x.to(model.params.device), y.to(model.params.device)
        assert not torch.isnan(x).any(), "Training input is Nan!"
        #self.learning_rate.optimizer.zero_grad()
        if get_embeddings:
            logits, preds, embs = model(x, return_embeddings=get_embeddings, stat_pooling=model.params.stat_pooling)
            assert not torch.isnan(logits).any(), "Training logits is Nan!"
            loss_value = self.loss(emb=embs, logits=logits, labels=y)
        else:
            logits, preds = model(x)
            assert not torch.isnan(logits).any(), "Training logits is Nan!"
            loss_value = self.loss(logits, y)
        assert not torch.isnan(loss_value), "Training loss is Nan!"
        assert loss_value > 0, "Training loss is negative!"
        loss_value.backward()

        return loss_value, preds

    def validation_step(self, model, x, y, get_embeddings=False):

        x, y = x.to(model.params.device), y.to(model.params.device)
        assert not torch.isnan(x).any(), "Validation input is Nan!"
        if get_embeddings:
            logits, preds, embs = model(x, return_embeddings=get_embeddings, stat_pooling=model.params.stat_pooling)
            assert not torch.isnan(logits).any(), "Validation logits is Nan!"
            loss_value = self.loss(emb=embs, logits=logits, labels=y)
        else:
            logits, preds = model(x)
            assert not torch.isnan(logits).any(), "Validation logits is Nan!"
            loss_value = self.loss(logits, y)
        assert not torch.isnan(loss_value), "Validation loss is Nan!"
        assert loss_value > 0, "Validation loss is negative!"

        return loss_value, preds

    def test_step(self, model, x, y):

        x, y = x.to(model.params.device), y.to(model.params.device)
        assert not torch.isnan(x).any(), "Test input is Nan!"
        logits, preds = model(x, stat_pooling=model.params.stat_pooling)
        assert not torch.isnan(logits).any(), "Test logits is Nan!"

        return preds


class EarlyStop:

    def __init__(self, tolerance, min_delta):
        """
        tolerance: number of overfitting epochs before break
        min_delta: variation threshold for the loss difference on the train and validation
        """
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.train_loss_prev = 0
        self.val_loss_prev = 0

    def __call__(self, train_loss, val_loss):
        if (train_loss <= self.train_loss_prev and self.val_loss_prev <= val_loss) or (
                self.train_loss_prev <= train_loss  and self.val_loss_prev <= val_loss): # #abs(val_loss - train_loss) >= self.min_delta:
            self.counter += 1
            self.train_loss_prev = train_loss
            self.val_loss_prev = val_loss
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            if self.counter > 0:
                self.counter -= 1


def encoder_gradual_unfreezer(model, use_lora=False, n=0):

    if n == 0: # Freeze whole encoder
        for p in model.backbone.base_model.encoder.parameters():
            if p.requires_grad: p.requires_grad = False
        custom_print("\nEncoder is frozen.")
        if use_lora:
            if model.__class__.__name__ == 'PeftModel':
                # Use PEFT
                module_idx = 0
                for module in model.backbone.base_model.encoder.modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        module.lora_A.requires_grad = False
                        module.lora_B.requires_grad = False
                        if hasattr(module, 'lora_dropout'):
                            module.lora_dropout.requires_grad = False
                        if hasattr(module, 'lora_embedding_A'):
                            module.lora_embedding_A.requires_grad = False
                            module.lora_embedding_B.requires_grad = False
                        custom_print(f"For {module_idx} module {module.__class__.__name__} LoRA adapter is frozen.")
                    module_idx += 1
            else:
                raise Exception("Not implemented. Use PEFT module for LoRA.")
        if hasattr(model, 'projector'):
            for p in model.projector.parameters():
                p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True
    else: # Unfreeze the last n layers
        total_layers = len(model.backbone.encoder.layers)
        if n > total_layers:
            custom_print("Warning: number of unfreezing layers > total number of layers! All encoder will be unfrozen.")
            n = total_layers
        layers_to_unfreeze = model.backbone.base_model.encoder.layers[-n:]
        for cur_layer in layers_to_unfreeze:
            for p in cur_layer.parameters():
                p.requires_grad = True
        # Unfreeze final LayerNorm
        for p in model.backbone.base_model.encoder.layer_norm.parameters():
            p.requires_grad = True
        custom_print(f"\n{n} upper layers of the encoder have been unfreezed.")

    return model


def last_epoch(train_loss_log, val_loss_log, lr_log, clustering_log, encoder_weights_log, epoch_fin, training_stage, save_model_dir):

    epoch_stop = len(val_loss_log)
    epoch_best = val_loss_log[:epoch_fin].index(min(val_loss_log[:epoch_fin]))+1
    custom_print(f"\nThe best model detected at epoch: {epoch_best}")
    # Plot and save the loss figure
    plot_loss_curves(train_loss_log, val_loss_log, epoch_best, epoch_stop, save_fig_to=save_model_dir+f"/Fig1_loss_{epoch_best}_{training_stage}.png", logscale=False)
    # Plot and save the learning rate figure
    plot_curve(lr_log, save_fig_to=save_model_dir+f"/Fig2_{training_stage}_lr_curve.png", title="Learning rate")
    # Plot and save the clustering measures figure
    if len(clustering_log) != 0: plot_curve(clustering_log, save_fig_to=save_model_dir+f"/Fig3_{training_stage}_clusterscores.png", title="Clustering scores")
    # Plot and save the encoder layers importance figure
    if len(encoder_weights_log) != 0: plot_curve(encoder_weights_log, save_fig_to=save_model_dir+f"/Fig4_{training_stage}_encoder_weights.png", title="Encoder weights")
    # Save log file
    save_log_to = save_model_dir + f"/training_logs_{training_stage}.txt"
    write_print_log(save_log_to)


def augment_batch(waveforms):

    augmented = []
    for cur_audio in waveforms:
        if torch.rand(1) > 0.2: # добавление шума
            cur_audio += torch.randn_like(cur_audio) * 0.005
            if torch.rand(1) > 0.5: # маскирование сигнала
                mask_len = torch.randint(100, 1000, (1,))
                start = torch.randint(0, cur_audio.shape[-1] - mask_len, (1,))
                cur_audio[..., start:start+mask_len] = 0.
        augmented.append(cur_audio)
    return torch.stack(augmented)


def model_train(model, trainer, dataloader):

    total_loss, total_correct = 0, 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        if trainer.params.augment:
            inputs = augment_batch(inputs)
        trainer.optimizer.zero_grad() # clear gradients
        cur_loss, cur_preds = trainer.training_step(model=model, x=inputs, y=labels, get_embeddings=(str(trainer.loss)[:-2]!='CrossEntropyLoss'))
        trainer.optimizer.step()
        total_loss += cur_loss.item()
        total_correct += (cur_preds.cpu() == labels).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy


@torch.no_grad()
def model_evaluate(model, trainer, dataloader, report_dict=False):

    model.eval()
    total_loss, total_correct = 0, 0
    cur_loss = 0
    all_preds, all_labels = [], []
    for inputs, labels in tqdm(dataloader, desc="Evaluating"):
        if report_dict:
            cur_preds = trainer.test_step(model, inputs, labels)
        else:
            cur_loss, cur_preds = trainer.validation_step(model, inputs, labels, get_embeddings=(str(trainer.loss)[:-2]!='CrossEntropyLoss'))
            total_loss += cur_loss.item()
        total_correct += (cur_preds.cpu() == labels).sum().item()
        all_preds.extend(cur_preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    clf_rep = classification_report(all_labels, all_preds, output_dict=report_dict)
    return avg_loss, accuracy, f1, clf_rep


def main(args):

    def _training_loop(train_params, model=None, load_modelname=None):
        """MAIN TRAINING LOOP"""

        if load_modelname is not None:
            custom_print(f"\nLoad from checkpoint: {load_modelname}")
            assert model is None
            args.loss_fn = 'cross_entropy'
            custom_print(f"\nApply {args.loss_fn} loss.")
            args.lr_scheduler = 'cos_annealing'
            custom_print(f"\nApply {args.lr_scheduler} LR scheduler.")
            #args.learning_rate *= 0.5
            checkpoint = torch.load(load_modelname, map_location=args.device)
            model = make_model(args)
            model.load_state_dict(checkpoint, strict=True)
            model.train()
            # Freeze whole model and unfreeze classifier
            custom_print("Freeze whole model except classifier.\n")
            if model.fusion_method is not None and model.fusion_method != 'mean':
                if isinstance(model.feature_fusion, nn.Parameter):
                    model.feature_fusion.requires_grad = False
                    n_param_ws = model.feature_fusion.numel()
                else: # GFF (nn.Linear)
                    for p in model.feature_fusion.parameters():
                        p.requires_grad = False
                    n_param_ws = sum(p.numel() for p in model.feature_fusion.parameters())
            else:
                n_param_ws = 0
            for p in model.backbone.parameters():
                p.requires_grad = False
            n_param_enc = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
            model.backbone.eval()
            for p in model.projector.parameters():
                p.requires_grad = False
            n_param_proj = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
            for p in model.classifier.parameters():
                p.requires_grad = True
            model.projector.eval()
            n_param_clf = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
            model.classifier.train()
            custom_print(f"Number of trainable parameters: {n_param_ws+n_param_enc+n_param_proj+n_param_clf}\n")
            custom_print("Model trainer update:")
        else:
            custom_print("Model trainer initialization:")
        trainer = ModelTrainer(model, args=args)
        overfit_checker = EarlyStop(tolerance=args.n_tolerance, min_delta=0.1)
        custom_print(f"\nStart training stage {train_params['training_stage']}\n")
        #print(f"Start LR: {trainer.scheduler.get_last_lr()[0]}")
        for epoch in range(args.num_epochs):
            # Create Datasets for full Dataframe splits
            df_train, df_val = get_stratified_data_splits(df_full, ratio_train=0.9, ratio_test=0.1)
            # Create Datasets for splits
            ds_train = EmotionDataset(args, df_train)
            ds_val = EmotionDataset(args, df_val)
            # Create Dataloaders for dataset splits
            train_loader = get_dataloader(ds_train, args.batch_size, args.n_workers)
            val_loader = get_dataloader(ds_val, args.batch_size, args.n_workers)

            # Visualize embeddings
            if load_modelname is None: # epoch % 2 == 0:
                print("\nEmbeddings distribution analysis...")
                sep_ratio, intra_dist = plot_embeddings_with_dataloader(
                    model, 
                    test_loader, # val_loader
                    args.save_model_dir + f"/embeddings_{train_params['training_stage']}_{epoch}.png", 
                    embedding_reducer, 
                    label_encoder
                )

                separation_log.append([sep_ratio, intra_dist])
            # Freezing encoder and LoRA
            if args.lora and args.n_tolerance * 2 <= len(train_loss_log):
                custom_print(f"\nFreeze encoder and LoRA adapters...")
                model = encoder_gradual_unfreezer(model, use_lora=args.lora)
                args.lora = False
                train_params["training_stage"] += 1

            custom_print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}, EPOCH {epoch+1}/{args.num_epochs}\n")
            # Training
            train_loss, train_acc = model_train(model, trainer, train_loader)
            assert not np.isnan(train_loss), f"Train loss is {train_loss}"
            cur_lr_clf = trainer.scheduler.get_last_lr()[-1]
            custom_print(f"\nTLoss: {train_loss:.4f} | TAcc: {train_acc:.4f} | LR Enc: {trainer.scheduler.get_last_lr()[0]} | LR Clf: {cur_lr_clf}\n")
            train_loss_log.append(train_loss)
            lr_log.append(cur_lr_clf)
            # Testing
            val_loss, val_acc, val_f1, _ = model_evaluate(model, trainer, test_loader, report_dict=False) # val_loader
            assert not np.isnan(val_loss), f"Validation loss is {val_loss}"
            custom_print(f"\nVLoss: {val_loss:.4f} | VAcc: {val_acc:.4f} | VF1: {val_f1:.4f}\n")
            val_loss_log.append(val_loss)

            # Сохранение лучшей модели и проверка на прерывание цикла обучения
            if train_params['training_stop'] is False and len(train_loss_log) > args.n_tolerance:
                grad_norms = []
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norms.append(p.grad.norm().item())
                if np.mean(grad_norms) < 1e-6:
                    print("\nGradient vanishing detected!\n")
                    train_params['training_stop'] = True

                if train_loss_log[-1] <= train_loss_log[-2] and val_loss_log[-1] <= val_loss_log[-2] and val_acc <= train_acc and train_loss_log[-1] <= val_loss_log[-1]:

                    if train_acc-val_acc <= 0.2 and val_loss_log[-1] < val_loss_log[-2]:

                        # Test the model
                        _, test_acc, test_f1, clf_report = model_evaluate(model, trainer, val_loader, report_dict=True) # test_loader
                        test_acc_log.append(test_acc)
                        custom_print(f"\nAcc: {test_acc:.4f} | F1: {test_f1:.4f}")
                        custom_print(f"\nClassification report test:\n{pd.DataFrame(clf_report).to_string(index=True)}")

                        f1_avg_test = torch.tensor([clf_report['macro avg']['f1-score']], dtype=torch.float32)
                        custom_print(f"\nAverage test F1-score: {f1_avg_test[0]:.2f}\n")
                        test_f1_log.append(f1_avg_test)

                        if max(test_f1_log) <= test_f1_log[-1]:
                            save_model_name = args.save_model_dir+f"/model_{train_params['training_stage']}_{epoch+1}.pt"
                            torch.save(model.state_dict(), save_model_name)
                            custom_print("Save improved model!\n")
                            train_params['count_no_improv'] = 0
                            epoch_best_idx_list.append(epoch)

                        # Trainig set evaluation
                        _, tmp_acc, tmp_f1, clf_report_tmp = model_evaluate(model, trainer, train_loader, report_dict=True)
                        custom_print(f"\nClassification report train:\n{pd.DataFrame(clf_report_tmp).to_string(index=True)}")

                        f1_avg_train = torch.tensor([clf_report_tmp['macro avg']['f1-score']], dtype=torch.float32)
                        custom_print(f"\nAverage train F1-score: {f1_avg_train[0]:.2f}\n")

                        if len(test_f1_log) > args.n_tolerance:
                            if test_f1_log[-1] < max(test_f1_log[-1-args.n_tolerance:-2]):
                                c = 0
                                for cur_f1 in test_f1_log[-1-args.n_tolerance:-1]:
                                    if test_f1_log[-1] < cur_f1:
                                        c += 1
                                    else:
                                        c -= 1
                                if c >= args.n_tolerance and val_loss_log[-1] > val_loss_log[-2]:
                                    train_params['epoch_stop'] = epoch+1
                                    custom_print(f"\nTest F1 doesn't improve over the {args.n_tolerance} epochs. Training stop epoch: {train_params['epoch_stop']}\n")
                                    train_params['training_stop'] = True

                        if min(val_loss_log[:-2]) < val_loss_log[-1]:
                            train_params['count_no_improv'] += 1
                        else:
                            if train_params['count_no_improv'] > 0: train_params['count_no_improv'] -= 1
                    else:
                        train_params['epoch_stop'] = epoch+1
                        custom_print(f"\nOverfitting by train-validation accuracy threshold detected: {train_params['epoch_stop']}\n")
                        train_params['training_stop'] = True # Overfitting detected (by threshold)!
                else:
                    train_params['count_no_improv'] += 1

                if not train_params['training_stop'] and train_params['count_no_improv'] > args.n_tolerance:
                    train_params['epoch_stop'] = epoch+1
                    custom_print(f"\nNo improvements (or very slow) during several epochs detected: {train_params['epoch_stop']}\n")
                    train_params['training_stop'] = True
                else:
                    train_params['count_no_improv'] -= 1

                if not train_params['training_stop'] and val_loss_log[0] < max(val_loss_log[-1-args.n_tolerance:-1]) and len(val_loss_log) > 2*args.n_tolerance:
                    c = 0
                    for cur_val_loss in val_loss_log[-1-args.n_tolerance:-1]:
                        if val_loss_log[0] < cur_val_loss:
                            c += 1
                    if c >= args.n_tolerance:
                        train_params['epoch_stop'] = epoch+1
                        custom_print(f"\nDivergence detected at epoch: {train_params['epoch_stop']}\n")
                        train_params['training_stop'] = True

                if not train_params['training_stop'] and len(test_acc_log) > args.n_tolerance:
                    if test_acc_log[-1] <= max(test_acc_log[-2-args.n_tolerance:-2]):
                        c = 0
                        for cur_test_acc in test_acc_log[-1-args.n_tolerance:-1]:
                            if test_acc_log[0] > cur_test_acc:
                                c += 1
                        if c >= args.n_tolerance and val_loss_log[-1] > val_loss_log[-2]:
                            train_params['epoch_stop'] = epoch+1
                            custom_print(f"\nTest Accuracy doesn't improve over the {args.n_tolerance} epochs. Training stop epoch: {train_params['epoch_stop']}\n")
                            train_params['training_stop'] = True
                else:
                    # Suboptimal convergence?
                    if train_loss_log[-1] <= train_loss_log[-2] and val_loss_log[-1] == val_loss_log[-2] == val_loss_log[-3]:
                        train_params['epoch_stop'] = epoch+1
                        custom_print(f"\nSuboptimal convergence detected at epoch: {train_params['epoch_stop']}\n")
                        train_params['training_stop'] = True
                    else:
                        # Overfitting?
                        overfit_checker(train_loss, val_loss)
                        if overfit_checker.early_stop:
                            train_params['epoch_stop'] = epoch+1
                            custom_print(f"\nOverfitting detected at epoch: {train_params['epoch_stop']}\n")
                            train_params['training_stop'] = True
            elif not train_params['training_stop'] and len(train_loss_log) > args.n_tolerance:
                if len(separation_log) > 1 and separation_log[-1][0] <= separation_log[-2][0]:
                    train_params['training_stop'] = True # switch to the encoder unfreezing stage
            else:
                pass # TODO?

            if train_params['training_stop']:
                if train_params['training_stop_counter'] == args.n_tolerance or not args.gradual_unfreezing: # switch to the final training stage
                    #last_epoch(train_loss_log, val_loss_log, lr_log, separation_log, epoch_fin, training_stage, args.save_model_dir)
                    break
                else: # reset hyperparameters for the new stage of encoder unfreezing
                    if len(separation_log) > 1 and separation_log[-1][0] - separation_log[-2][0] < 1e-3:  
                        #separation_log[-1] <= separation_log[-2]: # separation_log[-1] > np.mean(separation_log[:-2]):
                        custom_print(f"\nSeparation Ratio didn't improve:{separation_log[-1]}")
                        train_params['training_stop_counter'] += 1
                    if args.gradual_unfreezing:
                        train_params["training_stage"] += 1
                        model = encoder_gradual_unfreezer(model, n=train_params["training_stage"]) # gradual_unfreezer(model, n=(training_stop_counter + 1))
                        train_params['count_no_improv'] = 0
                        train_params['training_stop'] = False
                        overfit_checker = EarlyStop(tolerance=args.n_tolerance, min_delta=0.1)
                        custom_print("=" * 50)
                        custom_print(f"\nStart training stage {train_params['training_stage']}")
                    else: # на всякий случай
                        #last_epoch(train_loss_log, val_loss_log, lr_log, separation_log, epoch_fin, training_stage, args.save_model_dir)
                        break

            if args.lr_scheduler == "reduce_plateau":
                trainer.scheduler.step(val_loss)
            else:
                trainer.scheduler.step()
            if args.loss_fn == "adaptive_cosine_ce":
                custom_print(f"Update loss function params...")
                loss_alpha, loss_beta, loss_margin, loss_k_hard, loss_msg = trainer.loss.update_params()
                custom_print(f"{loss_msg}:\nalpha={loss_alpha:.4f}, beta={loss_beta:.4f}, margin={loss_margin:.4f}, k_hard={loss_k_hard:.4f}\n")
            if args.adapter == "hybrid_probing" or args.adapter == "linear_probing":
                if args.fusion_method is not None and args.fusion_method != "mean":
                    if args.fusion_method == "gff": # Gated Feature Fusion
                        with torch.no_grad():
                            enc_weights = torch.softmax(model.feature_fusion.weight.mean(dim=1), dim=0).detach().cpu().numpy()
                    elif args.fusion_method == "tws": # Trainable Weighted Sum
                        enc_weights = torch.nn.functional.softmax(model.feature_fusion, dim=0).detach().cpu().numpy()
                    if load_modelname is None and args.fusion_method == "tws":
                        # custom_print(f"Raw weights of trainable weighted sum \n{model.trainable_weighted_sum}")
                        custom_print(f"Softmax weights for the trainable weighted sum of encoder layers:\n")
                        for i, j in enumerate(enc_weights):
                            custom_print(f"Layer {i} importance: {j:.2f}\n")
                    enc_weights_log.append(enc_weights)
        # Final epoch
        last_epoch(train_loss_log, val_loss_log, lr_log, separation_log, enc_weights_log, epoch_fin, train_params['training_stage'], args.save_model_dir)
        save_model_name = args.save_model_dir+f"/model_{train_params['training_stage']}_fin.pt"
        torch.save(model.state_dict(), save_model_name)
        custom_print(f"\nTraining stage {train_params['training_stage']}: Save the final model!\n")

        return save_model_name, train_params

    def _undersample_group(group):
        return group.sample(n=min_class_size, replace=False, random_state=RANDOM_SEED)

    # MAIN ######################################################################################################################

    args.save_model_dir = 'exps/' + args.dataset_name + '/' + str(datetime.now()).replace(' ', '_').replace(':', '').split('.')[0]
    dataset_name = args.dataset_name.split('/')[-1]
    custom_print(f"Save results directory: {args.save_model_dir}")
    if not os.path.exists(args.save_model_dir): os.makedirs(args.save_model_dir)

    df = get_dataframe(args)

    plot_data_hist(
        data=df['emotion'].value_counts(), 
        title=f'"{dataset_name}" dataset distribution', 
        save_fig_to=args.save_model_dir)

    if args.w_classes:
        if 1: # Compute class weights
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(df['label']),
                y=df['label'].values)
            c_w = torch.tensor(class_weights, dtype=torch.float32).to(args.device)
        else: # Use constants
            if dataset_name == "iemocap" and args.num_classes == 4:
                c_w = torch.tensor([1.0, 1.5, 1.6, 1.0], dtype=torch.float32).to(args.device)
            else:
                raise Exception("Wrong class!")
        custom_print(f"\nClass weights for unbalanced data: {c_w}\n")
    else:
        c_w = None
    args.class_weights = c_w

    if args.balanced_data: # Dataset undersampling
        data_len_origin = df['emotion'].value_counts()
        min_class_size = df['emotion'].value_counts().min()
        df = df.groupby('emotion').apply(_undersample_group).reset_index(drop=True)
        custom_print("\nBalance dataset by min class...")
        for k, v in zip(df['emotion'].value_counts().keys(), df['emotion'].value_counts().values):
            custom_print(f"{k}: {v} files")
        data_len_cut = df['emotion'].value_counts()
        data_compare = pd.DataFrame({'Original': data_len_origin, 'Undersampled': data_len_cut})
        plot_data_hist(
            data=data_compare, 
            title=f'"{dataset_name}" balanced dataset distribution', 
            save_fig_to=args.save_model_dir)

    custom_print(f"\nClass distribution in full dataframe:\n{df['label'].value_counts().sort_index()}")
    # Create Datasets for full Dataframe splits
    df_full, df_test = get_stratified_data_splits(df)
    custom_print(f"\nClass distribution in train + val dataframe:\n{df_full['label'].value_counts().sort_index()}")
    custom_print(f"\nClass distribution in test dataframe:\n{df_test['label'].value_counts().sort_index()}")
    # Create Datasets for splits
    print("\nCreate test dataset...")
    ds_test = EmotionDataset(args, df_test)
    # Create Dataloaders for dataset splits
    test_loader = get_dataloader(ds_test, args.batch_size, args.n_workers) # get_dataloader(dataset_full, args.batch_size, args.n_workers)

    model = make_model(args)

    custom_print(f"\nTotal number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    if (args.freezing_encoder or args.gradual_unfreezing):
        assert not args.lora, "To freeze or unfreeze encoder manually, args.lora should be False!"
        model = encoder_gradual_unfreezer(model, use_lora=False)
    elif args.lora:
        assert not args.freezing_encoder, "To use LoRA, args.freezing_encoder should be False!"
        model.print_trainable_parameters()
    else:
        custom_print("Encoder is not frozen.\n")

    # Init training params
    epoch_fin = args.num_epochs
    epoch_best_idx_list = []
    train_loss_log, val_loss_log, test_acc_log, test_f1_log, lr_log, separation_log, enc_weights_log = [], [], [], [], [], [], []
    train_params = {
        "epoch_stop": None,
        "training_stop_counter": 0,
        "training_stop": False,
        "training_stage": 0,
        "count_no_improv": 0
    }

    # Init dimension reducer and label encoder for visualizations
    embedding_reducer = TSNE(n_components=2, random_state=RANDOM_SEED)
    label_encoder = LabelEncoder()

    # Run main phase of training
    custom_print("\nRUN TRAINING...\n")
    model.train()
    trained_modelname, train_params = _training_loop(train_params, model)

    if args.projector_out_dim is not None:
        # Reset train_params
        train_params["epoch_stop"] = None
        train_params["training_stop"] = False
        train_params["training_stop_counter"] = 0
        train_params["count_no_improv"] = 0
        train_params["training_stage"] += 1

        # Run 2nd phase of training
        custom_print("\nSECOND TRAINING PHASE...\n")
        trained_modelname, _ = _training_loop(train_params, load_modelname=trained_modelname)
        custom_print(f"\nFinal model: {trained_modelname}")

    return trained_modelname, train_params


if __name__ == "__main__":

    args = Hparams()
    args.dataroot = os.path.join(args.dataroot, args.dataset_name)
    assert args.num_epochs > 1

    print(f"Use device: {args.device}")
    custom_print(f"Dataset: {args.dataset_name}\n")
    custom_print("Training parameters:")
    custom_print(f"Base model: {args.model_name}\nTop k layers aggregation: {args.topk_layers}\nLoss: {args.loss_fn}\nBatch size: {args.batch_size}\nAugmentation: {args.augment}\n")
    saved_model_name = main(args)
    print(f"\nDONE!\nSaved model name: {saved_model_name}")
