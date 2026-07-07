import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, GINEConv, global_add_pool
from torchmetrics import MatthewsCorrCoef
from tqdm import tqdm

from .gpu_utils import get_device


@dataclass
class EnsemblePredictions:
    """
    Container for ensemble predictions.

    logits:
        Shape [num_models, num_atoms].

    y_trues:
        Shape [num_atoms].

    mol_ids:
        Shape [num_atoms].

    atom_ids:
        Shape [num_atoms].
    """

    logits: torch.Tensor
    y_trues: torch.Tensor
    mol_ids: torch.Tensor
    atom_ids: torch.Tensor
    descriptions: List[str]
    smiles: List[str]
    model_names: List[str]
    model_paths: List[str]

    def shannon_entropy(self, p: torch.Tensor) -> torch.Tensor:
        return -(
            p * torch.log2(p + 1e-14)
            + (1.0 - p) * torch.log2(1.0 - p + 1e-14)
        )

    def get_probabilities(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def get_ensemble_probability(self) -> torch.Tensor:
        return torch.mean(self.get_probabilities(), dim=0)

    def get_uncertainties(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = self.get_probabilities()
        mean_prob = torch.mean(probs, dim=0)

        u_tot = self.shannon_entropy(mean_prob)
        u_ale = torch.mean(self.shannon_entropy(probs), dim=0)
        u_epi = u_tot - u_ale

        return u_ale, u_epi, u_tot

    def to(self, device: torch.device) -> "EnsemblePredictions":
        return EnsemblePredictions(
            logits=self.logits.to(device),
            y_trues=self.y_trues.to(device),
            mol_ids=self.mol_ids.to(device),
            atom_ids=self.atom_ids.to(device),
            descriptions=self.descriptions,
            smiles=self.smiles,
            model_names=self.model_names,
            model_paths=self.model_paths,
        )

    def to_dataframe(self) -> pd.DataFrame:
        logits = self.logits.detach().cpu()
        probs = torch.sigmoid(logits)

        u_ale, u_epi, u_tot = self.get_uncertainties()
        u_ale = u_ale.detach().cpu()
        u_epi = u_epi.detach().cpu()
        u_tot = u_tot.detach().cpu()

        df = pd.DataFrame(
            {
                "mol_id": self.mol_ids.detach().cpu().numpy(),
                "atom_idx": self.atom_ids.detach().cpu().numpy(),
                "description": self.descriptions,
                "smiles": self.smiles,
                "y_true": self.y_trues.detach().cpu().numpy(),
                "ensemble_prob_mean": probs.mean(dim=0).numpy(),
                "ensemble_prob_std": probs.std(dim=0, unbiased=False).numpy(),
                "ensemble_logit_mean": logits.mean(dim=0).numpy(),
                "ensemble_logit_std": logits.std(dim=0, unbiased=False).numpy(),
                "uncertainty_aleatoric": u_ale.numpy(),
                "uncertainty_epistemic": u_epi.numpy(),
                "uncertainty_total": u_tot.numpy(),
            }
        )

        for i, model_name in enumerate(self.model_names):
            safe_name = (
                model_name.replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )

            df[f"{safe_name}_logit"] = logits[i].numpy()
            df[f"{safe_name}_prob"] = probs[i].numpy()

        return df


class GINEWithContextPooling(nn.Module):
    """
    GINEConv model with graph-level context pooling.

    Important:
        This class is kept architecture-compatible with existing checkpoints.
    """

    def __init__(
        self,
        params: Dict[str, int],
        hyperparams: Dict[str, Union[int, float]],
    ) -> None:
        super(GINEWithContextPooling, self).__init__()

        self.conv = nn.ModuleList()
        self.batch_norm = nn.ModuleList()

        in_channels: int = params["num_node_features"]
        out_channels: int = int(hyperparams["size_conv_layers"])

        for _ in range(int(hyperparams["num_conv_layers"])):
            self.conv.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(in_channels, out_channels),
                        BatchNorm(out_channels),
                        nn.LeakyReLU(),
                        nn.Linear(out_channels, out_channels),
                    ),
                    train_eps=True,
                    edge_dim=params["num_edge_features"],
                )
            )
            in_channels = out_channels
            self.batch_norm.append(BatchNorm(in_channels))

        mid_channels: int = int(hyperparams["size_final_mlp_layers"])

        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 2, mid_channels),
            BatchNorm(mid_channels),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(mid_channels, mid_channels),
            BatchNorm(mid_channels),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(mid_channels, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Keep the original forward logic unchanged.
        """

        x = data.x

        for i, (conv, batch_norm) in enumerate(zip(self.conv, self.batch_norm)):
            x = conv(x, data.edge_index, data.edge_attr)

            if i != len(self.conv) - 1:
                x = batch_norm(x)

            x = F.leaky_relu(x)

        x_pool = global_add_pool(x, data.batch)

        num_atoms_per_mol = torch.unique(
            data.batch,
            sorted=False,
            return_counts=True,
        )[1]

        x_pool_expanded = torch.repeat_interleave(
            x_pool,
            num_atoms_per_mol,
            dim=0,
        )

        x = torch.cat((x, x_pool_expanded), dim=1)

        x = self.classifier(x)

        return torch.flatten(x)

    @classmethod
    def get_params(
        cls,
        trial: optuna.trial.Trial,
    ) -> Dict[str, Union[int, float]]:
        learning_rate: float = trial.suggest_float(
            "learning_rate",
            1e-6,
            1e-3,
            log=True,
        )

        weight_decay: float = trial.suggest_float(
            "weight_decay",
            1e-5,
            1e-2,
            log=True,
        )

        pos_class_weight: float = trial.suggest_float(
            "pos_class_weight",
            2,
            3,
            log=False,
        )

        num_conv_layers: int = trial.suggest_int(
            "num_conv_layers",
            1,
            6,
            log=False,
        )

        size_conv_layers: int = trial.suggest_int(
            "size_conv_layers",
            low=64,
            high=1024,
            log=True,
        )

        size_final_mlp_layers: int = trial.suggest_int(
            "size_final_mlp_layers",
            low=64,
            high=1024,
            log=True,
        )

        return {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "pos_class_weight": pos_class_weight,
            "num_conv_layers": num_conv_layers,
            "size_conv_layers": size_conv_layers,
            "size_final_mlp_layers": size_final_mlp_layers,
        }


class SOMPredictor(nn.Module):
    """Graph Neural Network for site-of-metabolism prediction."""

    def __init__(
        self,
        data_params: Dict[str, int],
        hyperparams: Dict[str, Union[int, float]],
    ) -> None:
        super().__init__()

        self.device = get_device()

        self.model = GINEWithContextPooling(data_params, hyperparams)
        self.model.to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                hyperparams["pos_class_weight"],
                dtype=torch.float32,
            ).to(self.device)
        )

        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
        )

        self.mcc = MatthewsCorrCoef(task="binary").to(self.device)

        self.hyperparams = hyperparams
        self.data_params = data_params

    def forward(self, batch: Data) -> torch.Tensor:
        batch = batch.to(self.device)
        return self.model(batch)

    def train_step(
        self,
        batch: Data,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Tuple[float, float]:
        self.train()
        self.optimizer.zero_grad()

        if scaler is not None and torch.cuda.is_available():
            with torch.amp.autocast(device_type="cuda"):
                logits = self(batch)
                loss = self.loss_fn(logits, batch.y.float())

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

        else:
            logits = self(batch)
            loss = self.loss_fn(logits, batch.y.float())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        mcc = self.mcc(torch.sigmoid(logits), batch.y)

        return loss.item(), mcc.item()

    def val_step(self, batch: Data) -> Tuple[float, float]:
        self.eval()

        with torch.no_grad():
            logits = self(batch)
            loss = self.loss_fn(logits, batch.y.float())
            mcc = self.mcc(torch.sigmoid(logits), batch.y)

        return loss.item(), mcc.item()

    @staticmethod
    def _expand_graph_metadata_to_atoms(
        batch: Data,
        field_name: str,
        default_value: str = "",
    ) -> List[str]:
        num_nodes = int(batch.num_nodes)

        if not hasattr(batch, "batch"):
            return [default_value for _ in range(num_nodes)]

        graph_indices = batch.batch.detach().cpu().tolist()

        values = getattr(batch, field_name, None)

        if values is None:
            return [default_value for _ in range(num_nodes)]

        if isinstance(values, str):
            values = [values]

        if not isinstance(values, (list, tuple)):
            values = [str(values)]

        values = [str(v) for v in values]

        expanded: List[str] = []
        for graph_idx in graph_indices:
            if 0 <= graph_idx < len(values):
                expanded.append(values[graph_idx])
            else:
                expanded.append(default_value)

        return expanded

    def predict(
        self,
        batch: Data,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[str],
        List[str],
    ]:
        self.eval()

        batch = batch.to(self.device)

        with torch.no_grad():
            logits = self(batch)

        descriptions_atom = self._expand_graph_metadata_to_atoms(
            batch=batch,
            field_name="description",
            default_value="",
        )

        smiles_atom = self._expand_graph_metadata_to_atoms(
            batch=batch,
            field_name="smiles",
            default_value="",
        )

        return (
            logits.detach().cpu(),
            batch.y.detach().cpu(),
            batch.mol_id.detach().cpu(),
            batch.atom_id.detach().cpu(),
            descriptions_atom,
            smiles_atom,
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "hyperparams": self.hyperparams,
                "data_params": self.data_params,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "SOMPredictor":
        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(
                path,
                map_location="cpu",
            )
        except Exception:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

        model = cls(
            checkpoint["data_params"],
            checkpoint["hyperparams"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            try:
                model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass

        return model

    def fit(
        self,
        train_loader: DataLoader[Data],
        val_loader: Optional[DataLoader[Data]] = None,
        max_epochs: int = 500,
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        patience: int = 20,
    ) -> int:
        writer = SummaryWriter(log_dir) if log_dir else None

        best_val_loss = float("inf")
        patience_counter = 0
        actual_epochs = 0

        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        for epoch in tqdm(range(max_epochs)):
            actual_epochs = epoch + 1

            train_losses, train_mccs = [], []

            for batch in train_loader:
                loss, mcc = self.train_step(batch, scaler)
                train_losses.append(loss)
                train_mccs.append(mcc)

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_train_mcc = sum(train_mccs) / len(train_mccs)

            if val_loader:
                val_losses, val_mccs = [], []

                for batch in val_loader:
                    loss, mcc = self.val_step(batch)
                    val_losses.append(loss)
                    val_mccs.append(mcc)

                avg_val_loss = sum(val_losses) / len(val_losses)
                avg_val_mcc = sum(val_mccs) / len(val_mccs)

                if writer:
                    writer.add_scalar("train/loss", avg_train_loss, epoch)
                    writer.add_scalar("train/mcc", avg_train_mcc, epoch)
                    writer.add_scalar("val/loss", avg_val_loss, epoch)
                    writer.add_scalar("val/mcc", avg_val_mcc, epoch)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0

                    if checkpoint_dir:
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        self.save(os.path.join(checkpoint_dir, "best_model.ckpt"))

                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            else:
                if writer:
                    writer.add_scalar("train/loss", avg_train_loss, epoch)
                    writer.add_scalar("train/mcc", avg_train_mcc, epoch)

                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    self.save(os.path.join(checkpoint_dir, "best_model.ckpt"))

        if writer:
            writer.close()

        return actual_epochs


def _infer_model_name(path: str, index: int) -> str:
    p = os.path.normpath(path)
    parts = p.split(os.sep)

    for part in reversed(parts):
        if part.startswith("model_"):
            return part

    return f"model_{index:02d}"


def predict_ensemble(
    data: DataLoader[Data],
    model_paths: List[str],
) -> EnsemblePredictions:
    models = [SOMPredictor.load(path) for path in model_paths]
    model_names = [_infer_model_name(path, i) for i, path in enumerate(model_paths)]

    all_logits: List[torch.Tensor] = []

    y_trues: Optional[torch.Tensor] = None
    mol_ids: Optional[torch.Tensor] = None
    atom_ids: Optional[torch.Tensor] = None
    descriptions: Optional[List[str]] = None
    smiles: Optional[List[str]] = None

    with torch.no_grad():
        for i, model in enumerate(models):
            print(f"Predicting with {model_names[i]} [{i + 1}/{len(models)}]")

            logits_list: List[torch.Tensor] = []
            y_trues_list: List[torch.Tensor] = []
            mol_ids_list: List[torch.Tensor] = []
            atom_ids_list: List[torch.Tensor] = []
            descriptions_list: List[str] = []
            smiles_list: List[str] = []

            for batch in data:
                (
                    pred_logits,
                    pred_y,
                    pred_mol,
                    pred_atom,
                    pred_desc,
                    pred_smiles,
                ) = model.predict(batch)

                logits_list.append(pred_logits)
                y_trues_list.append(pred_y)
                mol_ids_list.append(pred_mol)
                atom_ids_list.append(pred_atom)
                descriptions_list.extend(pred_desc)
                smiles_list.extend(pred_smiles)

            model_logits = torch.cat(logits_list, dim=0)
            model_y_trues = torch.cat(y_trues_list, dim=0)
            model_mol_ids = torch.cat(mol_ids_list, dim=0)
            model_atom_ids = torch.cat(atom_ids_list, dim=0)

            all_logits.append(model_logits)

            if i == 0:
                y_trues = model_y_trues
                mol_ids = model_mol_ids
                atom_ids = model_atom_ids
                descriptions = descriptions_list
                smiles = smiles_list

    assert y_trues is not None
    assert mol_ids is not None
    assert atom_ids is not None
    assert descriptions is not None
    assert smiles is not None

    ensemble_logits = torch.stack(all_logits, dim=0)

    return EnsemblePredictions(
        logits=ensemble_logits,
        y_trues=y_trues,
        mol_ids=mol_ids,
        atom_ids=atom_ids,
        descriptions=descriptions,
        smiles=smiles,
        model_names=model_names,
        model_paths=model_paths,
    )
