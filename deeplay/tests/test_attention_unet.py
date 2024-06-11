import unittest
import torch

from deeplay import AttentionUNet


def positional_encoding(t, emb_dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, emb_dim, 2).float() / emb_dim))
    inv_freq = inv_freq.to(t.device)
    pos_enc_a = torch.sin(t.repeat(1, emb_dim // 2) * inv_freq)
    pos_enc_b = torch.cos(t.repeat(1, emb_dim // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc


class TestAttentionUNet(unittest.TestCase):
    ...

    def test_attn_unet_ddpm(self):

        attn_unet = AttentionUNet(
            in_channels=1,
            channels=[8, 16, 32],
            base_channels=[64, 64],
            channel_attention=[True, True, True],
            out_channels=1,
            position_embedding_dim=64,
        )
        attn_unet.build()

        # Test on a batch of 2
        x = torch.rand(2, 1, 64, 64)
        t = torch.rand(2, 1)
        output = attn_unet(x, positional_encoding(t, 64), y=None, context=None)

        # Check output shape
        self.assertEqual(output.shape, (2, 1, 64, 64))

    def test_ddpm_without_channel_attention(self):

        attn_unet = AttentionUNet(
            in_channels=1,
            channels=[8, 16, 32],
            base_channels=[64, 64],
            channel_attention=[False, False, False],
            out_channels=1,
            position_embedding_dim=64,
        )
        attn_unet.build()

        # Test on a batch of 2
        x = torch.rand(2, 1, 64, 64)
        t = torch.rand(2, 1)
        output = attn_unet(x, positional_encoding(t, 64), y=None, context=None)

        # Check output shape
        self.assertEqual(output.shape, (2, 1, 64, 64))

    def test_attention_unet_conditional_ddpm(self):

        attn_unet = AttentionUNet(
            in_channels=1,
            channels=[8, 16, 32],
            base_channels=[64, 64],
            channel_attention=[True, True, True],
            out_channels=1,
            position_embedding_dim=64,
            num_classes=10,
        )
        attn_unet.build()

        # Test on a batch of 2
        x = torch.rand(2, 1, 64, 64)
        t = torch.rand(2, 1)
        y = torch.randint(0, 10, (2,))
        output = attn_unet(x, positional_encoding(t, 64), y=y, context=None)

        # Check output shape
        self.assertEqual(output.shape, (2, 1, 64, 64))

    def test_attention_unet_context_ddpm(self):

        attn_unet = AttentionUNet(
            in_channels=1,
            channels=[8, 16, 32],
            base_channels=[64, 64],
            channel_attention=[True, True, True],
            out_channels=1,
            position_embedding_dim=64,
            context_embedding_dim=64,  # 768
        )
        attn_unet.build()

        # Test on a batch of 2
        x = torch.rand(2, 1, 64, 64)
        t = torch.rand(2, 1)
        context = torch.rand(2, 77, 64)
        output = attn_unet(x, positional_encoding(t, 64), y=None, context=context)

        # Check output shape
        self.assertEqual(output.shape, (2, 1, 64, 64))

    def test_context_without_channel_attention(self):

        attn_unet = AttentionUNet(
            in_channels=1,
            channels=[8, 16, 32],
            base_channels=[64, 64],
            channel_attention=[False, False, False],
            out_channels=1,
            position_embedding_dim=64,
            context_embedding_dim=64,
        )
        attn_unet.build()

        # Test on a batch of 2
        x = torch.rand(2, 1, 64, 64)
        t = torch.rand(2, 1)
        context = torch.rand(2, 77, 64)
        output = attn_unet(x, positional_encoding(t, 64), y=None, context=context)

        # Check output shape
        self.assertEqual(output.shape, (2, 1, 64, 64))

    def test_self_attention_heads(self):

        attn_unet = AttentionUNet(
            in_channels=1,
            channels=[8, 16, 32],
            base_channels=[64, 64],
            channel_attention=[True, True, True],
            out_channels=1,
            position_embedding_dim=64,
            num_attention_heads={"self": 2, "cross": 2},
        )
        attn_unet.build()

        # Test on a batch of 2
        x = torch.rand(2, 1, 64, 64)
        t = torch.rand(2, 1)
        output = attn_unet(x, positional_encoding(t, 64), y=None, context=None)

        # Check output shape
        self.assertEqual(output.shape, (2, 1, 64, 64))
