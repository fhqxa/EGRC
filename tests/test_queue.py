import unittest
from run_main import build_jobs


class QueueTests(unittest.TestCase):
    def test_default_eight_jobs_and_three_seed_selection(self):
        jobs = build_jobs()
        self.assertEqual(len(jobs), 8)
        self.assertEqual(len({job["args"].exp for job in jobs}), 8)
        for job in jobs:
            args = job["args"]
            self.assertEqual(args.edge_loss_weight, 0.01 if args.shot == 1 else 0.05)
            self.assertEqual(args.backbone_chunk_size, 32)
        jobs = build_jobs(datasets=["fc100"], shots=[1], seeds=[42, 43, 44])
        self.assertEqual(len(jobs), 3)
        self.assertEqual([j["args"].train_episode_seed for j in jobs], [42, 43, 44])
        self.assertEqual({j["args"].val_episode_seed for j in jobs}, {100042})


if __name__ == "__main__":
    unittest.main()
