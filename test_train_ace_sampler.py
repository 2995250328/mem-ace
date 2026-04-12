import tempfile
import unittest
from pathlib import Path

import train_ace_sampler as train_sampler


class TrainACESamplerPathTest(unittest.TestCase):
    def test_resolve_sampler_output_defaults_to_ace_sampler_04_evaluation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene = Path(tmpdir) / '7Scenes' / 'stairs'
            scene.mkdir(parents=True)
            requested_output = Path('stairs_sampler.pt')

            run_dir, output_path = train_sampler.resolve_sampler_output_path(scene, requested_output)

            self.assertTrue(output_path.name.endswith('stairs_sampler.pt'))
            self.assertTrue(str(run_dir).endswith('ace_sampler/04_evaluation/7Scenes/stairs'))
            self.assertEqual(output_path.parent, run_dir)


if __name__ == '__main__':
    unittest.main()
