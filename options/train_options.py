from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--g_lr', type=float, default=0.0004, help='initial learning rate for adam')
        self.parser.add_argument('--d_lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--g_beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--d_beta1', type=float, default=0.0, help='momentum term of adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--lambda_B2', type=float, default=1.0, help='WEight term for BCE after processing e.g. outside log()')
        self.parser.add_argument('--lambda_C', type=float, default=10.0, help='weight for WGAN gradient penalty')
        self.parser.add_argument('--lambda_D', type=float, default=10.0, help='weight for folder predictor')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--identity', type=float, default=0.5,
                                 help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss.'
                                      'For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.parser.add_argument('--local_loss', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--local_critic', action='store_true', help='Whether to use critic only on masked region, or global')
        self.parser.add_argument('--num_discrims', type=int, default=5, help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--high_iter', type=int, default=25, help='Number of critic iterations at beginning of training')
        self.parser.add_argument('--low_iter', type=int, default=5, help='Number of critic iterations after initial phase')
        self.parser.add_argument('--no_mask_to_critic', action='store_true', help="Don't pass mask to discriminator/critic")
        self.parser.add_argument('--diff_in_numerator', action='store_true', help='Use frequency diff in numerator when creating weight mask')
        self.parser.add_argument('--weighted_grad', action='store_true', help='Apply weight mask to gradient')
        self.parser.add_argument('--optim_type', type=str, default='adam', help='Type of optimiser to use adam|rmsprop')
        self.parser.add_argument('--alpha', type=float, default=0.99, help="Smoothing value for rms prop")
        self.parser.add_argument('--use_hinge', action='store_true', help='Use hinge loss with critic')
        self.parser.add_argument('--with_BCE', action='store_true', help='Include BCE loss')
        self.parser.add_argument('--log_BCE', action='store_true', help='Apply log to BCE loss')
        self.parser.add_argument('--log_L2', action='store_true', help='Apply log to MSE')
        self.isTrain = True
