from typing import Optional

from deeplay.activelearning.strategies.strategy import Strategy
from deeplay.activelearning.data import ActiveLearningDataset, JointDataset
from deeplay.activelearning.criterion import ActiveLearningCriterion, Margin
from deeplay.module import DeeplayModule
from deeplay.external.optimizers import Adam


import torch
import torch.nn.functional as F


class AdversarialStrategy(Strategy):

    def __init__(
        self,
        backbone: DeeplayModule,
        classification_head: DeeplayModule,
        discriminator_head: DeeplayModule,
        train_pool: ActiveLearningDataset,
        val_pool: Optional[ActiveLearningDataset] = None,
        test: Optional[torch.utils.data.Dataset] = None,
        criterion: ActiveLearningCriterion = Margin(),
        uncertainty_weight: float = 0.3,
        discriminator_weight: float = 0.7,
        gradient_penalty_weight: float = 0.02,
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        backbone_optimizer: Optional[torch.optim.Optimizer] = None,
        classification_head_optimizer: Optional[torch.optim.Optimizer] = None,
        discriminator_head_optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ):
        super().__init__(
            train_pool,
            val_pool,
            test,
            batch_size,
            val_batch_size,
            test_batch_size,
            **kwargs
        )

        self.backbone = backbone
        self.classification_head = classification_head
        self.discriminator_head = discriminator_head
        self.uncertainty_criterion = criterion
        # assert 0 <= uncertainty_weight <= 1, f"uncertainty_weight must be in [0, 1], got {uncertainty_weight}"
        self.uncertainty_weight = uncertainty_weight
        self.discriminator_weight = discriminator_weight
        self.gradient_penalty_weight = gradient_penalty_weight
        self.automatic_optimization = False

        self.backbone_optimizer = backbone_optimizer or Adam(lr=1e-3)
        self.classification_head_optimizer = classification_head_optimizer or Adam(
            lr=1e-3
        )
        self.discriminator_head_optimizer = discriminator_head_optimizer or Adam(
            lr=1e-3
        )

    def query_strategy(self, pool, n):
        """Implement the query strategy here."""
        self.eval()
        X = pool.get_unannotated_samples()

        latents = self.backbone.predict(X)
        probs = self.classification_head.predict(latents).softmax(dim=1)
        dis_score = self.discriminator_head.predict(latents).flatten()

        uncertainly_score = self.uncertainty_criterion.score(probs)
        total_score = (
            self.uncertainty_weight * uncertainly_score
            + self.discriminator_weight * dis_score
        )

        return total_score.sort()[1][:n]

    def training_step(self, batch, batch_idx):
        self.train()
        _, label_x, label_y, unlabel_x, _ = batch

        opt_fea, opt_clf, opt_dis = self.optimizers()

        lb_z = self.backbone(label_x)
        unlb_z = self.backbone(unlabel_x)

        opt_fea.zero_grad()
        opt_clf.zero_grad()
        lb_out = self.classification_head(lb_z)

        pred_loss = torch.mean(F.cross_entropy(lb_out, label_y))

        # mse disc loss
        unlab_disc = self.discriminator_head(unlb_z).pow(2).mean()
        lab_disc = (self.discriminator_head(lb_z) - 1).pow(2).mean()
        wae_loss = pred_loss - 1e-3 * (unlab_disc + lab_disc)

        wae_loss.backward()
        opt_fea.step()
        opt_clf.step()

        gradient_penalty = self.gradient_penalty(lb_z.detach(), unlb_z.detach())

        unlab_disc = self.discriminator_head(unlb_z.detach()).pow(2).mean()
        lab_disc = (self.discriminator_head(lb_z.detach()) - 1).pow(2).mean()
        dis_loss = (
            unlab_disc + lab_disc + gradient_penalty * self.gradient_penalty_weight
        )
        opt_dis.zero_grad()
        dis_loss.backward()
        opt_dis.step()

        self.log(
            "loss", wae_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "disc_loss",
            dis_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "pred_loss",
            pred_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "gradient_penalty",
            gradient_penalty,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return wae_loss

    def train_dataloader(self):
        annotated_data = self.train_pool.get_annotated_data()
        unannotated_data = self.train_pool.get_unannotated_data()
        X_1, Y_1 = zip(*[(x, y) for x, y in annotated_data])
        X_2, Y_2 = zip(*[(x, y) for x, y in unannotated_data])
        data = JointDataset(X_1, Y_1, X_2, Y_2)
        return torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=True
        )

    def configure_optimizers(self):

        backbone_optimizer = self.create_optimizer_with_params(
            self.backbone_optimizer, self.backbone.parameters()
        )
        classification_head_optimizer = self.create_optimizer_with_params(
            self.classification_head_optimizer, self.classification_head.parameters()
        )
        discriminator_head_optimizer = self.create_optimizer_with_params(
            self.discriminator_head_optimizer, self.discriminator_head.parameters()
        )

        return [
            backbone_optimizer,
            classification_head_optimizer,
            discriminator_head_optimizer,
        ]

    def forward(self, x):
        return self.classification_head(self.backbone(x))

    def gradient_penalty(self, real, fake):
        alpha = torch.rand(real.size(0), 1).to(real.device)
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
        disc_interpolates = self.discriminator_head(interpolates).flatten()
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
