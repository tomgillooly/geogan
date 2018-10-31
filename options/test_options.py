from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--metrics', type=str, default="", help='Which metrics to apply to output images')
        self.parser.add_argument('--visualise_hausdorff', action='store_true', help='Show related pixels used to calculate Hausdorff distance')
        self.parser.add_argument('--visualise_ot', action='store_true', help='Show which pixel maps to which under optimal transport')
        self.parser.add_argument('--no_images', action='store_true')
        self.parser.add_argument('--images_only', action='store_true')
        self.parser.add_argument('--start_index', type=int, default=0, help="Which data series to load first")
        self.parser.add_argument('--end_index', type=int, default=-1, help="Which data series to test up to, default is how_many")
        self.parser.add_argument('--test_repeats', type=int, default=5, help="How many times to test series with same mask location - only used in exhaustive tests")
        self.parser.add_argument('--mask_overlap_thresh', type=float, default=0.3, help="What percentage masks can overlap by, only used in exhaustive test")
        self.isTrain = False
