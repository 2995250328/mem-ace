from pathlib import Path


def _strtobool(v):
    return str(v).lower() in ('true', '1', 'yes')


def add_sampler_train_args(parser):
    """Phase 1: train SamplerNet e2e."""
    g = parser.add_argument_group('sampler_train')
    g.add_argument('scene', type=Path, help='scene root (contains train/)')
    g.add_argument('ace_head', type=Path, help='trained ACE head .pt')
    g.add_argument('sampler_output', type=Path, help='output sampler .pt')
    g.add_argument('--encoder_path', type=Path, default=Path('ace_encoder_pretrained.pt'))
    g.add_argument('--num_head_blocks', type=int, default=1)
    g.add_argument('--use_homogeneous', type=_strtobool, default=False)
    g.add_argument('--image_resolution', type=int, default=480)
    g.add_argument('--use_aug', type=_strtobool, default=True)
    g.add_argument('--aug_rotation', type=int, default=15)
    g.add_argument('--aug_scale', type=float, default=1.5)
    g.add_argument('--sampler_epochs', type=int, default=5)
    g.add_argument('--sampler_lr', type=float, default=1e-3)
    g.add_argument('--sampler_alpha', type=float, default=0.1,
                   help='confidence target: exp(-alpha * repro_error_px)')
    g.add_argument('--use_half', type=_strtobool, default=True)
    g.add_argument('--device', type=str, default='cuda')
    g.add_argument('--use_mc_dropout', type=_strtobool, default=False,
                   help='enable MC Dropout uncertainty estimation in Phase 1')
    g.add_argument('--mc_samples', type=int, default=10,
                   help='number of MC Dropout forward passes per image')
    g.add_argument('--mc_dropout_p', type=float, default=0.1,
                   help='dropout probability for UncertaintyHead')
    g.add_argument('--sampler_beta', type=float, default=0.01,
                   help='weight for MC uncertainty term: exp(-(alpha*error + beta*var))')
    g.add_argument('--uncertainty_head_path', type=Path, default=None,
                   help='pretrained UncertaintyHead .pt; None = train from scratch')


def add_sampler_buffer_args(parser):
    """Phase 2: buffer filling options."""
    g = parser.add_argument_group('sampler_buffer')
    g.add_argument('--sampler_path', type=Path, default=None,
                   help='trained SamplerNet .pt; None = original random sampling')
    g.add_argument('--sampler_ratio', type=float, default=0.7,
                   help='fraction of samples from sampler top-k (rest random)')
    g.add_argument('--use_neighbors', type=_strtobool, default=False,
                   help='expand sampler selections to 4-connected neighbors')
