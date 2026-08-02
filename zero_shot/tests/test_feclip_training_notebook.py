import json
import unittest
from pathlib import Path


class FECLIPTrainingNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook_path = Path(__file__).parents[1] / "kaggle_train_feclip.ipynb"
        cls.notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_paper_training_configuration_is_present(self):
        expected_fragments = (
            "IMAGE_SIZE = 336",
            "EPOCHS = 9",
            "LEARNING_RATE = 5e-4",
            "TOTAL_BATCH_SIZE = 16",
            "FFE_WINDOW = 3",
            "LFS_WINDOW = 3",
            "FREQUENCY_LAMBDA = 0.1",
            "NUM_STAGES = 4",
            "torch.optim.Adam",
        )
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)

    def test_paper_prompts_and_cross_dataset_protocol_are_explicit(self):
        self.assertIn("A photo of a normal object", self.source)
        self.assertIn("A photo of a damaged object", self.source)
        self.assertIn("MVTec-trained weights evaluate VisA", self.source)
        self.assertIn("VisA-trained weights evaluate MVTec", self.source)
        self.assertIn("'intended_target': 'MVTec' if TRAIN_DATASET == 'VisA'", self.source)

    def test_frequency_adapters_and_paper_fusion_are_implemented(self):
        self.assertIn("class FFEAdapter(nn.Module)", self.source)
        self.assertIn("class LFSAdapter(nn.Module)", self.source)
        self.assertIn("frequency.mean(dim=(2, 3))", self.source)
        self.assertIn(
            "self.frequency_lambda * frequency + (1.0 - self.frequency_lambda) * patches",
            self.source,
        )

    def test_losses_are_averaged_across_four_stages(self):
        self.assertIn("F.binary_cross_entropy", self.source)
        self.assertIn("focal_loss(probs, masks) + dice_loss", self.source)
        self.assertIn("loss_cls = torch.stack(class_losses).mean()", self.source)
        self.assertIn("loss_mask = torch.stack(mask_losses).mean()", self.source)

    def test_dual_t4_ddp_preserves_paper_total_batch_size(self):
        self.assertIn("NUM_TRAINING_PROCESSES = 2", self.source)
        self.assertIn("Accelerator(gradient_accumulation_steps=accumulation", self.source)
        self.assertIn("'torch.distributed.run'", self.source)
        self.assertIn("f'--nproc_per_node={NUM_TRAINING_PROCESSES}'", self.source)
        self.assertNotIn("notebook_launcher(", self.source)
        self.assertIn(
            "TOTAL_BATCH_SIZE // (NUM_TRAINING_PROCESSES * GPU_MICRO_BATCH_SIZE)",
            self.source,
        )

    def test_torchrun_workers_do_not_fork_or_pickle_notebook_cuda_state(self):
        self.assertIn("training_script_source", self.source)
        self.assertIn("feclip_torchrun_worker.py", self.source)
        self.assertIn("worker_path.write_text(training_script_source", self.source)
        self.assertIn("subprocess.run(launch_command", self.source)
        self.assertNotIn("cloudpickle", self.source)

    def test_parent_runtime_configuration_is_passed_to_torchrun_workers(self):
        self.assertIn("FECLIP_RUNTIME_CONFIG_JSON", self.source)
        self.assertIn("'TRAIN_DATASET': TRAIN_DATASET", self.source)
        self.assertIn("TRAIN_DATASET = runtime_config['TRAIN_DATASET']", self.source)
        self.assertIn("launch_environment['FECLIP_RUNTIME_CONFIG_JSON']", self.source)

    def test_unreleased_code_limitations_are_not_hidden(self):
        self.assertIn("independent implementation, not official author code", self.source)
        self.assertIn("paper_underspecified_choices", self.source)
        self.assertIn("learnable_patch_fc_optimization", self.source)


if __name__ == "__main__":
    unittest.main()
