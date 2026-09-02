import pytest
param = pytest.mark.parametrize

import torch
from vector_quantize_pytorch import VectorQuantize

@param('training', (True, False))
def test_topk_and_manual_ema_update(training):

    vq1 = VectorQuantize(
        dim = 256,
        codebook_size = 512
    )

    vq2 = VectorQuantize(
        dim = 256,
        codebook_size = 512
    )

    vq2.load_state_dict(vq1.state_dict())

    x = torch.randn(1, 1024, 256)

    mask = torch.randint(0, 2, (1, 1024)).bool() if training else None

    vq1.train(training)
    quantize1, indices1, commit_loss1 = vq1(x, mask = mask)

    vq2.train(training)
    quantize2, indices2, commit_losses = vq2(x, mask = mask, topk = 1, ema_update = False)

    assert quantize2.shape == (1, 1024, 1, 256)
    assert indices2.shape == (1, 1024, 1)
    assert commit_losses.shape == (1, 1024, 1)

    top_quantize2 = quantize2[..., 0, :]
    top_indices2 = indices2[..., 0]

    assert torch.equal(indices1, top_indices2)
    assert torch.allclose(quantize1, top_quantize2)

    if training:

        assert torch.allclose(commit_loss1, commit_losses.sum() / mask.sum())

        assert not torch.allclose(vq1._codebook.embed_avg, vq2._codebook.embed_avg)

        vq2.update_ema_indices(x, top_indices2, mask = mask)

        assert torch.allclose(vq1._codebook.cluster_size, vq2._codebook.cluster_size)
        assert torch.allclose(vq1._codebook.embed_avg, vq2._codebook.embed_avg)
        assert torch.allclose(vq1.codebook, vq2.codebook)

@param('codebook_dim', (256, 128))
@param('training', (True, False))
def test_beam_search(
    codebook_dim,
    training
):
    from vector_quantize_pytorch import ResidualVQ

    residual_vq = ResidualVQ(
        dim = 256,
        codebook_dim = codebook_dim,
        num_quantizers = 8,      # specify number of quantizers
        codebook_size = 1024,    # codebook size
        quantize_dropout = True,
        beam_size = 2,
        eval_beam_size = 3
    )

    residual_vq.train(training)

    x = torch.randn(4, 1024, 256).requires_grad_(training)

    for _ in range(5):
        quantized, indices, commit_loss = residual_vq(x)

    assert quantized.shape == (4, 1024, 256)
    assert indices.shape == (4, 1024, 8)
    assert commit_loss.shape == (8,)

@param('training', (True, False))
def test_topk_single_token(training):

    vq = VectorQuantize(dim = 16, codebook_size = 8)
    vq.train(training)

    quantize, indices, loss = vq(torch.randn(2, 16), topk = 2)

    assert quantize.shape == (2, 2, 16)
    assert indices.shape == (2, 2)
    assert loss.shape == (2, 2)

@param('training', (True, False))
def test_topk_channel_first(training):

    vq = VectorQuantize(dim = 16, codebook_size = 8, channel_last = False)
    vq.train(training)

    quantize, indices, loss = vq(torch.randn(2, 16, 4), topk = 2)

    assert quantize.shape == (2, 16, 4, 2)
    assert indices.shape == (2, 4, 2)
    assert loss.shape == (2, 4, 2)

@param('training', (True, False))
def test_topk_image_fmap(training):

    vq = VectorQuantize(dim = 16, codebook_size = 8, accept_image_fmap = True)
    vq.train(training)

    quantize, indices, loss = vq(torch.randn(2, 16, 4, 4), topk = 2)

    assert quantize.shape == (2, 16, 4, 4, 2)
    assert indices.shape == (2, 4, 4, 2)
    assert loss.shape == (2, 4, 4, 2)

def test_topk_with_explicit_indices_and_transform_fn():
    from einops import repeat

    vq = VectorQuantize(dim = 16, codebook_size = 8)

    x = torch.randn(2, 4, 16)
    indices = torch.randint(0, 8, (2, 4))
    transform_fn = lambda embed: repeat(embed, 'h c d -> h b n c d', b = 2, n = 4)

    quantize, loss = vq(x, indices = indices, topk = 2, codebook_transform_fn = transform_fn)

    assert quantize.shape == (2, 4, 2, 16)
    assert loss.ndim == 0

def test_topk_auto_cast_and_clamp():

    vq = VectorQuantize(dim = 16, codebook_size = 8).eval()

    quantize, indices, _ = vq(torch.randn(2, 4, 16), topk = 2.0)
    assert indices.shape == (2, 4, 2)

    quantize, indices, _ = vq(torch.randn(2, 4, 16), topk = 9)
    assert indices.shape == (2, 4, 8)

@param('training', (True, False))
def test_beam_search_with_batched_mask(training):

    from vector_quantize_pytorch import ResidualVQ

    residual_vq = ResidualVQ(
        dim = 8,
        codebook_dim = 4,
        num_quantizers = 3,
        codebook_size = 16,
        beam_size = 2,
        eval_beam_size = 2
    )

    residual_vq.train(training)

    x = torch.randn(2, 5, 8).requires_grad_(training)
    mask = torch.tensor([
        [True, True, True, False, False],
        [True, True, False, False, False]
    ])

    quantized, indices, commit_loss = residual_vq(x, mask = mask)

    assert quantized.shape == x.shape
    assert indices.shape == (2, 5, 3)
    assert commit_loss.shape == (3,)
    assert torch.isfinite(quantized).all()
    assert torch.isfinite(commit_loss).all()
    assert torch.equal(indices[~mask], torch.full_like(indices[~mask], -1))
