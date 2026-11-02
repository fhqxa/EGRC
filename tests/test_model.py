import unittest
import torch
from egrc.model import EGRCConfig, EGRCHead, egrc_training_loss


class ModelTests(unittest.TestCase):
    def test_default_edge_weight_matches_the_shot_setting(self):
        for shot, weight in ((1, 0.01), (5, 0.05)):
            head = EGRCHead(
                EGRCConfig(dim=16, relation_dim=8, relation_hidden_dim=16, task_hidden_dim=8)
            )
            _, aux = head(torch.randn(3 * shot, 16), torch.randn(6, 16), 3, shot)
            labels = torch.arange(3).repeat(2)
            node_labels = torch.cat([torch.arange(3).repeat(shot), labels])
            state = torch.get_rng_state()
            actual, _ = egrc_training_loss(aux["graph_logits"], labels, aux, node_labels)
            torch.set_rng_state(state)
            expected, _ = egrc_training_loss(
                aux["graph_logits"], labels, aux, node_labels, edge_loss_weight=weight
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_one_and_five_shot_are_finite_and_trainable_without_query_label_input(self):
        for shot in (1, 5):
            with self.subTest(shot=shot):
                torch.manual_seed(42)
                config = EGRCConfig(
                    dim=16, relation_dim=8, relation_hidden_dim=16, task_hidden_dim=8
                )
                head = EGRCHead(config)
                support = torch.randn(3 * shot, 16, requires_grad=True)
                query = torch.randn(6, 16, requires_grad=True)
                _, aux = head(support, query, way=3, shot=shot)
                logits = aux["graph_logits"]
                self.assertEqual(tuple(logits.shape), (6, 3))
                self.assertTrue(torch.isfinite(logits).all())
                labels = torch.arange(3).repeat(2)
                node_labels = torch.cat([torch.arange(3).repeat(shot), labels])
                loss, _ = egrc_training_loss(
                    logits,
                    labels,
                    aux,
                    node_labels,
                    edge_loss_weight=0.01 if shot == 1 else 0.05,
                    edge_kl_weight=0.5,
                )
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(support.grad).all())
                self.assertTrue(torch.isfinite(query.grad).all())
                self.assertGreater(head.edge_builder.output[-1].weight.grad.abs().sum(), 0)
                self.assertTrue(torch.isfinite(head.task_encoder.network[0].weight.grad).all())
                self.assertGreater(head.edge_builder.task_film.weight.grad.abs().sum(), 0)


if __name__ == "__main__":
    unittest.main()
