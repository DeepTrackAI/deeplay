import io
from typing import Iterable, List, Tuple, Type
import unittest
import torch

from deeplay.applications.application import Application
from deeplay.external.optimizers.adam import Adam
from deeplay.trainer import Trainer


class BaseApplicationTest:

    class BaseTest(unittest.TestCase):

        def get_class(self) -> Type[Application]:
            raise NotImplementedError

        def get_networks(self) -> Iterable[Application]:
            raise NotImplementedError

        def get_training_data(self) -> Iterable[Tuple[torch.Tensor, ...]]:
            raise NotImplementedError

        def get_validation_data(self) -> Iterable[Tuple[torch.Tensor, ...]]:
            return self.get_training_data()

        def get_testing_data(self) -> Iterable[Tuple[torch.Tensor, ...]]:
            return self.get_training_data()

        def get_predict_data(self) -> Iterable[Tuple[torch.Tensor, ...]]:
            for data in self.get_training_data():
                yield (data[0],)

        def get_expected_train_history_keys(self) -> Iterable[List[str]]:
            for network in self.get_networks():
                metrics = network.train_metrics
                metric_names = [str(k) + "_epoch" for k in metrics.keys()]
                yield ["train_loss_epoch"] + metric_names

        def get_expected_val_history_keys(self) -> Iterable[List[str]]:
            for network in self.get_networks():
                metrics = network.val_metrics
                metric_names = [str(k) + "_epoch" for k in metrics.keys()]
                yield ["val_loss_epoch"] + metric_names

        def get_expected_step_history_keys(self) -> Iterable[List[str]]:
            for network in self.get_networks():
                metrics = network.train_metrics
                metric_names = [str(k) + "_step" for k in metrics.keys()]
                yield ["train_loss_step"] + metric_names

        def make_dataset(self, data: Tuple[torch.Tensor, ...]):
            return torch.utils.data.TensorDataset(*data)

        def make_dataloader(self, data: Tuple[torch.Tensor, ...]):
            dataset = torch.utils.data.TensorDataset(*data)
            return torch.utils.data.DataLoader(dataset, batch_size=2)

        def train_model_with_trainer(
            self, model: Application, training_data: Tuple[torch.Tensor, ...]
        ):
            dataloader = self.make_dataloader(training_data)
            trainer = Trainer(
                max_epochs=2,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            trainer.fit(model, dataloader)
            return model

        def train_model_with_trainer_and_validate(
            self,
            model: Application,
            training_data: Tuple[torch.Tensor, ...],
            validation_data: Tuple[torch.Tensor, ...],
        ):
            dataloader = self.make_dataloader(training_data)
            val_dataloader = self.make_dataloader(validation_data)

            trainer = Trainer(
                max_epochs=2,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            trainer.fit(model, dataloader, val_dataloader)
            return model

        def test_can_build(self):
            for network in self.get_networks():
                network.build()

        def test_can_create(self):
            for network in self.get_networks():
                network.create()

        def test_build_and_create_has_same_trainable_parameters(self):
            for network in self.get_networks():
                created = network.create()
                built = network.build()

                self.assertEqual(
                    sum(p.numel() for p in created.parameters()),
                    sum(p.numel() for p in built.parameters()),
                )

        def test_build_and_create_has_same_modules(self):
            for network in self.get_networks():
                created = network.create()
                built = network.build()

                for c, b in zip(created.modules(), built.modules()):
                    self.assertEqual(type(c), type(b))
                    for c_p, b_p in zip(c.parameters(), b.parameters()):
                        self.assertEqual(c_p.shape, b_p.shape)

        def test_can_configure_optimizer(self):
            for network in self.get_networks():
                network.configure(optimizer=Adam(lr=1))
                created = network.create()
                optimizer = created.configure_optimizers()
                self.assertIsInstance(optimizer, torch.optim.Adam)
                self.assertEqual(optimizer.param_groups[0]["lr"], 1)

        def test_does_train_with_trainer(self):
            for network, training_data, train_epoch_keys, step_keys in zip(
                self.get_networks(),
                self.get_training_data(),
                self.get_expected_train_history_keys(),
                self.get_expected_step_history_keys(),
            ):
                network = self.train_model_with_trainer(network.create(), training_data)

                self.assertEqual(
                    set(network.trainer.history.history.keys()),
                    set(train_epoch_keys),
                )
                self.assertEqual(
                    set(network.trainer.history.step_history.keys()),
                    set(step_keys),
                )

        def test_does_train_with_trainer_and_validate(self):
            for (
                network,
                training_data,
                validation_data,
                train_epoch_keys,
                val_epoch_keys,
                train_step_keys,
            ) in zip(
                self.get_networks(),
                self.get_training_data(),
                self.get_validation_data(),
                self.get_expected_train_history_keys(),
                self.get_expected_val_history_keys(),
                self.get_expected_step_history_keys(),
            ):
                network = self.train_model_with_trainer_and_validate(
                    network.create(), training_data, validation_data
                )
                self.assertEqual(
                    set(network.trainer.history.history.keys()),
                    set(train_epoch_keys + val_epoch_keys),
                )
                self.assertEqual(
                    set(network.trainer.history.step_history.keys()),
                    set(train_step_keys),
                )

        def test_does_train_with_fit(self):
            for network, training_data in zip(
                self.get_networks(), self.get_training_data()
            ):
                history = network.fit(training_data, max_epochs=1, batch_size=2)

        def test_does_train_with_fit_dataset(self):
            for network, training_data in zip(
                self.get_networks(), self.get_training_data()
            ):
                dataset = self.make_dataset(training_data)
                history = network.fit(dataset, max_epochs=1, batch_size=2)

        def test_does_train_with_fit_and_validate(self):
            for network, training_data, validation_data in zip(
                self.get_networks(),
                self.get_training_data(),
                self.get_validation_data(),
            ):
                history = network.fit(
                    training_data,
                    val_data=validation_data,
                    max_epochs=1,
                    batch_size=2,
                    val_batch_size=2,
                )

        def test_can_save_and_load_statedict(self):
            for network, data in zip(self.get_networks(), self.get_training_data()):
                net = network.create()

                file = io.BytesIO()
                torch.save(net.state_dict(), file)
                file.seek(0)

                out = net(data[0].to(net.device))

                net2 = network.create()
                net2.load_state_dict(torch.load(file))

                out2 = net2(data[0].to(net2.device))

                self.assertTrue(torch.allclose(out, out2))

        def test_can_save_and_load_checkpoint(self):
            for network, data in zip(self.get_networks(), self.get_training_data()):
                model = network.create()
                dataloader = self.make_dataloader(data)
                trainer = Trainer(
                    max_epochs=1,
                    enable_checkpointing=True,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                )
                trainer.fit(model, dataloader)

                model2 = type(model).load_from_checkpoint(
                    trainer.checkpoint_callback.best_model_path
                )
