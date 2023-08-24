import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')

    # general
    parser.add_argument('--data_dir', type=str, help='the directory for the video sequence')
    parser.add_argument('--expname', type=str, default='', help='experiment name')
    parser.add_argument('--local_rank', type=int, default=0, help='rank for distributed training')
    parser.add_argument('--save_dir', type=str, default='out/', help='output dir')
    parser.add_argument('--ckpt_path', type=str, default='', help='checkpoint path')
    parser.add_argument('--no_reload', action='store_true', help='do not reload the weights')
    parser.add_argument('--distributed', type=int, default=0, help='if use distributed training')
    parser.add_argument('--num_iters', type=int, default=100000, help='number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--load_opt', type=int, default=1, help='if loading optimizers')
    parser.add_argument('--load_scheduler', type=int, default=1, help='if loading schedulers')
    parser.add_argument('--loader_seed', type=int, default=12,
                        help='the random seed used for DataLoader')

    # data
    parser.add_argument('--dataset_types', type=str, default='flow', help='only flow is included in the current version')
    parser.add_argument('--dataset_weights', nargs='+', type=float, default=[1.], help='the weight for each dataset')
    parser.add_argument('--num_imgs', type=int, default=250, help='max number of images to train')
    parser.add_argument('--num_pairs', type=int, default=8, help='# image pairs to sample in each batch')
    parser.add_argument('--num_pts', type=int, default=256, help='# pts to sample from each pair of images')

    # lr
    parser.add_argument('--lr_feature', type=float, default=1e-3, help='learning rate for feature mlp')
    parser.add_argument('--lr_deform', type=float, default=1e-4, help='learning rate for deform mlp')
    parser.add_argument('--lr_color', type=float, default=3e-4, help='learning rate for color mlp')
    parser.add_argument("--lrate_decay_steps", type=int, default=20000,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--lrate_decay_factor", type=float, default=0.5,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--grad_clip", type=float, default=0, help='clip the gradient to avoid training instability')

    # network training
    parser.add_argument('--use_error_map', action='store_true', help='use error map')
    parser.add_argument('--use_count_map', action='store_true', help='use count map')
    parser.add_argument('--use_affine', action='store_true',
                        help='if using additional 2D affine transformation layers for x, y in the invertible network')
    parser.add_argument('--mask_near', action='store_true',
                        help='if mask out the nearest samples in the beginning of the optimization,'
                             'may be helpful to avoid bad initialization associated with wrong surface ordering'
                             'e.g., a surface is initialized at very small depth but should instead be farther away')
    parser.add_argument('--num_samples_ray', type=int, default=32, help='number of samples per ray')
    parser.add_argument('--pe_freq', type=int, default=4, help='the freq for pe used in the affine coupling layers')
    parser.add_argument('--min_depth', type=float, default=0, help='the minimum depth value')
    parser.add_argument('--max_depth', type=float, default=2, help='the maximum depth value')
    parser.add_argument('--start_interval', type=int, default=20, help='the starting interval')
    parser.add_argument('--max_padding', type=float, default=0,
                        help='if predicted pixel locs exceed this padding, mask them out for training')

    # inference
    parser.add_argument('--chunk_size', type=int, default=10000, help='chunk size for rendering depth and rgb')
    parser.add_argument('--use_max_loc', action='store_true',
                        help='during inference, if using only the sample with maximum blending weight on the ray'
                             'to compute correspondence. If set to False, the correspondences will be computed'
                             'the same way as training, i.e., compositing all samples along the ray.')
    parser.add_argument('--query_frame_id', type=int, default=0, help='the id of the query frame')
    parser.add_argument('--vis_occlusion', action='store_true',
                        help='if marking occluded pixels as crosses for visualization')
    parser.add_argument('--occlusion_th', type=float, default=0.99,
                        help='to determine if a mapped 3d location in the target frame is occluded or not,'
                             ' we look at the fraction of light absorbed by samples in front of this location '
                             'on the ray in the target frame (i.e., 1 - transmittance)'
                             'if that value is higher than this threshold, the mapped point is considered as occluded')
    parser.add_argument('--foreground_mask_path', type=str, default='',
                        help='providing the path for foreground mask file for generating trails')

    # log
    parser.add_argument('--i_print', type=int, default=100, help='frequency for printing losses')
    parser.add_argument('--i_img', type=int, default=500, help='frequency for writing visualizations to tensorboard')
    parser.add_argument('--i_weight', type=int, default=20000, help='frequency for saving ckpts')
    parser.add_argument('--i_cache', type=int, default=20000, help='frequency for caching current flow predictions')

    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

    args = parser.parse_args()
    return args



