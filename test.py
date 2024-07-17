import logging
import math
import yaml
import unittest
import torch


def getLogger(cls):
    logging.basicConfig(
        level=logging.INFO,
        format="[{cls}] %(message)s".format(cls=cls),
    )
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    log = logging.getLogger(cls)
    log.addHandler(console)
    return log


class TestCaseWrapper(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)


class EncoderTest(TestCaseWrapper):
    cfg = yaml.safe_load(open("config/vae.yaml", "r"))

    def test_encoder_output(self):
        from module.vae import Encoder

        x = torch.randn(2, 3, self.cfg["img_shape"], self.cfg["img_shape"])
        encoder = Encoder(self.cfg)
        out: torch.tensor = encoder(x)
        # print("output", out.shape)
        # print("DEBUG: latent window size:", int(math.sqrt(encoder.fc_inp/512)))

        assert out.size() == (2, 1024)


class DecoderTest(TestCaseWrapper):
    cfg = yaml.safe_load(open("config/vae.yaml", "r"))

    def test_decoder_output(self):
        # print("test_decoder_output")
        from module.vae import Decoder

        decoder = Decoder(self.cfg, 512 * 59 * 59)
        x = torch.randn(2, self.cfg["model"]["latent_dim"])
        # print("decoder output:",decoder(x).size())
        assert decoder(x).size() == (2, 3, 256, 256)


class AutoEncoderTest(TestCaseWrapper):
    cfg = yaml.safe_load(open("config/vae.yaml", "r"))

    def test_autoencoder_output(self):
        from module.vae import AutoEncoder

        model = AutoEncoder(self.cfg)
        img_tensor = torch.randn(2, 3, self.cfg["img_shape"], self.cfg["img_shape"])
        out = model(img_tensor)
        assert (
            out.size() == img_tensor.size()
        ), "model output ({}) not matched with given image tensor ({})".format(
            out.size(), img_tensor.size()
        )


def main():
    unittest.main()


if __name__ == "__main__":
    main()
