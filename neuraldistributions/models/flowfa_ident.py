from warnings import warn
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import LowRankMultivariateNormal

from ..utility import set_random_seed
from nnfabrik.utility.nnf_helper import split_module_name, dynamic_import
from nnfabrik.utility.nn_helpers import set_random_seed
from neuralpredictors.training import eval_state
from . import transforms
from .transforms import (
    Identity,
    SQRT,
    Anscombe,
    ELU,
    InvELU,
    ELUF,
    Affine,
    Log,
    Flow,
    MeanTransfom,
    Exp,
    tanh_or_shift,
    Softplus,
    LeakyReLU,
    LowRankAffine
)


def get_learned_transforms(name, n_dimensions=1):
    if name == "learned-mini":
        return [
            Affine(n_dimensions=n_dimensions, only_positive_shift=True),
            Log(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
            Exp(n_dimensions=n_dimensions),
        ]
    elif name == "learned-leaky-rand-init":
        normal = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1e-3]))
        return [
            Affine(n_dimensions=n_dimensions, init_a=normal.sample((1, n_dimensions))[0].T,
                   init_t=normal.sample((1, n_dimensions))[0].T),
            ELU(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions, init_a=normal.sample((1, n_dimensions))[0].T,
                   init_t=normal.sample((1, n_dimensions))[0].T),
            ELU(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions, init_a=normal.sample((1, n_dimensions))[0].T,
                   init_t=normal.sample((1, n_dimensions))[0].T),
            ELU(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions, init_a=normal.sample((1, n_dimensions))[0].T,
                   init_t=normal.sample((1, n_dimensions))[0].T),
        ]

    elif name == "learned-leaky":
        return [
            Affine(n_dimensions=n_dimensions),
            ELU(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
            ELU(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
            ELU(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions, ),
        ]

    elif name == "learned-leaky-low-rank-single":
        return [
            Identity(n_dimensions=n_dimensions),
            LowRankAffine(n_dimensions=n_dimensions, rank=n_dimensions),
            Identity(n_dimensions=n_dimensions),
        ]

    elif name.startswith("learned-leaky-low-rank-k-"):
        rank = int(name.split("learned-leaky-low-rank-k-")[1])
        return [
            Identity(n_dimensions=n_dimensions),
            LowRankAffine(n_dimensions=n_dimensions, rank=rank),
            ELU(n_dimensions=n_dimensions),
            LowRankAffine(n_dimensions=n_dimensions, rank=rank),
            ELU(n_dimensions=n_dimensions),
            LowRankAffine(n_dimensions=n_dimensions, rank=rank),
            ELU(n_dimensions=n_dimensions),
            LowRankAffine(n_dimensions=n_dimensions, rank=rank),
            Identity(n_dimensions=n_dimensions)
        ]

    elif name == "learned2":
        return [
            Affine(n_dimensions=n_dimensions, only_positive_shift=True),
            Log(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
            ELU(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
            ELU(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
            Exp(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
        ]
    elif name == "learned2_with_tanh":
        return [
            tanh_or_shift(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions, only_positive_shift=True),
            Exp(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
            ELU(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
            ELU(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
            Exp(n_dimensions=n_dimensions),
            Affine(n_dimensions=n_dimensions),
        ]

    elif name == "learned6":
        return [
            Affine(n_dimensions=n_dimensions, only_positive_shift=True),
            InvELU(n_dimensions=n_dimensions, offset=1.0),
            Affine(n_dimensions=n_dimensions),
            ELU(n_dimensions=n_dimensions),
            Affine(
                n_dimensions=n_dimensions,
                init_t=1.0,
                init_a=1.0,
                learn_t=False,
                learn_a=False,
            ),  # go back to +ve support with elu+1
            Affine(
                n_dimensions=n_dimensions, only_positive_shift=True
            ),  # allow shifts but stay in the +ve support
            InvELU(
                n_dimensions=n_dimensions, offset=1.0
            ),  # go to continuous space with inverse elu+1
            Affine(n_dimensions=n_dimensions),
            ELU(n_dimensions=n_dimensions),
            Affine(
                n_dimensions=n_dimensions,
                init_t=1.0,
                init_a=1.0,
                learn_t=False,
                learn_a=False,
            ),
            Affine(
                n_dimensions=n_dimensions,
                only_positive_shift=True,
                init_t=1.0,
            ),  # allow shifts but stay in the +ve support
            InvELU(
                n_dimensions=n_dimensions, offset=1.0
            ),  # go to continuous space with inverse elu+1
        ]


def freeze_params(model):
    for name, param in model.named_parameters():
        param.requires_grad_(False)


def unfreeze_params(model):
    for name, param in model.named_parameters():
        param.requires_grad_(True)


class FlowFA_Ident(nn.Module):
    def __init__(
            self,
            dataloaders,
            seed,
            image_model_fn,
            image_model_config,
            d_latent,
            use_avg_reg,
            latent_weights_sparsity_reg_lambda,
            sample_transform,
            mean_transform,
            per_neuron_samples_transform,
            init_psi_diag_coef,
            init_C_coef,
            unit_variance_constraint,
    ):
        super().__init__()

        # set the random seed
        set_random_seed(seed)

        try:
            module_path, class_name = split_module_name(image_model_fn)
            model_fn = dynamic_import(module_path, class_name)
        except:
            raise ValueError("model function does not exist.")

        self.encoding_model = model_fn(dataloaders, seed, **image_model_config)
        self.d_latent = None if d_latent == 0 else d_latent
        self.use_avg_reg = use_avg_reg
        self.latent_weights_sparsity_reg_lambda = latent_weights_sparsity_reg_lambda
        self.unit_variance_constraint = unit_variance_constraint

        dataloaders = dataloaders["train"] if "train" in dataloaders else dataloaders
        temp_b = next(iter(list(dataloaders.values())[0]))._asdict()
        d_out = temp_b["targets"].shape[1]
        self.d_out = d_out

        if self.d_latent is not None:
            self._C = nn.Parameter(
                torch.rand(d_latent, d_out) * init_C_coef, requires_grad=False
            )

        self.per_neuron_samples_transform = per_neuron_samples_transform
        n_dimensions = d_out if per_neuron_samples_transform else 1

        if sample_transform is None:
            self.sample_transform = getattr(transforms, "identity")(numpy=False)

        elif sample_transform in ["identity", "sqrt", "anscombe"] + [
            f"example{i + 1}" for i in range(10)
        ]:
            self.sample_transform = getattr(transforms, sample_transform)(numpy=False)

        elif "learned" in sample_transform:
            self.sample_transform = Flow(
                get_learned_transforms(sample_transform, n_dimensions=n_dimensions)
            )

        else:
            raise ValueError("The passed sample_transform is not available.")

        if (mean_transform is None) or (mean_transform == "identity"):
            self.mean_transform = lambda x: x
        elif mean_transform == "learned":
            self.mean_transform = MeanTransfom(hidden_layers=2, hidden_features=10)
        elif mean_transform == "anscombe":
            self.mean_transform = lambda x: Anscombe.anscombe(x) - 1 / (4 * x.sqrt())
        else:
            raise ValueError(
                "At the moment only three options for sample transform are available: identity, learned and anscombe"
            )

    @property
    def sigma(self):
        return torch.eye(1000)

    def forward(self, *args, data_key=None, return_all=False):

        image_model_pred = self.encoding_model(*args, data_key=data_key) + 1e-8
        mu = self.mean_transform(image_model_pred)
        return (mu, self.sigma) if return_all else mu

    def predict_mean(self, *batch, data_key=None):
        return self.forward(*batch, data_key=data_key)

    def log_likelihood(self, *batch, data_key=None, in_bits=False):

        # get model predictions with the covariance matrix
        mu = self.forward(*batch, data_key=data_key)

        inputs, targets = batch[:2]
        transformed_targets, logdet = self.sample_transform(targets)

        var = torch.eye(mu.shape[1]).reshape(1, mu.shape[1], mu.shape[1]).to(mu.device)
        dist = torch.distributions.MultivariateNormal(mu, var)
        loglikelihood = dist.log_prob(transformed_targets) + logdet.sum(dim=1)

        return loglikelihood / np.log(2.0) if in_bits else loglikelihood

    def loss(self, *batch, data_key=None, use_avg=False):
        agg_fn = torch.mean if use_avg else torch.sum
        loss = -self.log_likelihood(*batch, data_key=data_key)

        return agg_fn(loss)

    def regularizer(self, data_key=None):
        return self.encoding_model.regularizer(data_key)

    def apply_changes_while_training(self):
        return

    def evaluate(self):
        raise NotImplementedError()


def flowfa_ident(
        dataloaders,
        seed,
        image_model_fn=None,
        image_model_config=None,
        d_latent=0,
        use_avg_reg=False,
        latent_weights_sparsity_reg_lambda=0.0,
        sample_transform=None,
        mean_transform=None,
        per_neuron_samples_transform=False,
        init_psi_diag_coef=0.01,
        init_C_coef=0.1,
        unit_variance_constraint=False,
):
    if image_model_fn is None:
        raise ValueError("Please specify image-model function.")

    if image_model_config is None:
        raise ValueError(
            "Please specify the config of the image-model, excluding dataloaders and seed."
        )

    device = image_model_config.get("device", "cuda")
    set_random_seed(seed)
    return FlowFA_Ident(
        dataloaders,
        seed,
        image_model_fn=image_model_fn,
        image_model_config=image_model_config,
        d_latent=d_latent,
        use_avg_reg=use_avg_reg,
        latent_weights_sparsity_reg_lambda=latent_weights_sparsity_reg_lambda,
        sample_transform=sample_transform,
        mean_transform=mean_transform,
        per_neuron_samples_transform=per_neuron_samples_transform,
        init_psi_diag_coef=init_psi_diag_coef,
        init_C_coef=init_C_coef,
        unit_variance_constraint=unit_variance_constraint,
    ).to(device)
