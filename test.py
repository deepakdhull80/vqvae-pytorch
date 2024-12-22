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
        out, _ = model(img_tensor)
        assert (
            out.size() == img_tensor.size()
        ), "model output ({}) not matched with given image tensor ({})".format(
            out.size(), img_tensor.size()
        )


class VQVAETest(TestCaseWrapper):
    from module.vqvae.vqvae import VQVAE

    device = "mps"
    cfg = yaml.safe_load(open("config/vqvae.yaml", "r"))
    model = VQVAE(cfg)
    model.to(device)
    IMG_SIZE = 256

    def test_vqvae_output(self):
        img = torch.randint(0, 256, size=(1, 3, self.IMG_SIZE, self.IMG_SIZE)).to(
            self.device
        )
        img = img / 255.0
        x, (codebook_loss, reconstruction_loss), vq_cfg = self.model(img)
        print(x.shape, codebook_loss, reconstruction_loss, vq_cfg["perplexity"])
        loss = reconstruction_loss + codebook_loss
        loss.backward()


def main():
    unittest.main()


if __name__ == "__main__":
    main()
