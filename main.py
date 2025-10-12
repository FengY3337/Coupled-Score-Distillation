import argparse

from utils import *
from GaussianSplatting.provider import GSDataset
from GaussianSplatting.GS_no_network import Renderer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # default parameters
    parser.add_argument('--test', type=bool, default=False, help="Test mode, evaluate the results of 3D-GS and mesh")
    parser.add_argument('--test_model_path', type=str, default='A_results/frog.pth', help='save path of the tested file')
    parser.add_argument('--workspace', type=str, default='stage_1/', help='test results save path')
    parser.add_argument('--name', type=str, default='Submitted_', help='experiment name')

    #### dmtet parameters (optimize mesh)
    # geometry-15000(dmtet_iters)-5000(t5)；texture-3000(dmtet_iters)-1000(t5)
    parser.add_argument('--dmtet_iters', type=int, default=15000, help='iters for optimizing geometry or texture of mesh')
    parser.add_argument('--dmtet_t5_iters', type=int, default=5000, help='iters for annealing to t_max = 500')
    parser.add_argument('--MLP_initial_iters', type=int, default=1000, help='iters for initializing MLP of texture')
    parser.add_argument('--tet_grid_size', type=int, default=256, help='dmtet size')

    parser.add_argument('--GS_dmtet', type=bool, default=False, help='optimize dmtet')
    parser.add_argument('--dmtet_init_path', type=str, default='./dmtet_initial/stage_1_results/frog.ply',
                        help='optimized 3D-GS path for initializing dmtet')

    parser.add_argument('--dmtet_finetune', type=bool, default=False,
                        help='optimize texture of dmtet mesh (True), otherwise for geometry (False)')
    parser.add_argument('--dmtet_init_path_scale', type=str, default='./dmtet_initial/stage_2_results/frog_scaled.ply',
                        help='scaled 3D-GS for initializing texture of mesh')
    parser.add_argument('--finetune_path', type=str, default='./dmtet_initial/stage_2_results/frog.pth',
                        help='dmtet mesh file path for fine-tuning texture')

    parser.add_argument('--dmtet_lr', type=float, default=5e-3, help='learning rate for optimizing dmtet')
    parser.add_argument('--lambda_normal', type=float, default=5000, help='loss scale for mesh normal smoothness')
    parser.add_argument('--lambda_lap', type=float, default=0.5, help='loss scale for mesh laplacian')
    parser.add_argument('--density_thresh', default=0.2, help='density thresh for mesh extraction')

    parser.add_argument('--albedo_lr', type=float, default=1e-3, help='learning rate for optimizing texture')
    parser.add_argument('--num_layers', type=int, default=1, help='layer number of MLP for predicting albedo value')
    parser.add_argument('--hidden_dim', type=int, default=32, help='hidden dims of MLP')
    parser.add_argument('--hash_resolution', type=int, default=2048, help='resolution of hashgrid')

    #### training options
    parser.add_argument('--text', default='A frog lying on a lily pad.', help="text prompt")
    parser.add_argument('--text_short', default='frog', help="short form of text prompt")
    parser.add_argument('--negative_text', default='', type=str, help="negative text prompt")
    parser.add_argument('--checkpoint_stage1', type=str, default=None, help='load point cloud for 3D-GS')
    parser.add_argument('--use_tensorboardX', type=bool, default=True, help='adopt tensorboardX')

    parser.add_argument('--iters', type=int, default=4000, help="total iters for 3D-GS")
    parser.add_argument('--seed', type=int, default=None, help='seed')
    parser.add_argument('--iter_per_loader', type=int, default=50, help="sample number of each dataloader")
    parser.add_argument('--val_interval', type=int, default=5, help="validate per dataloader")
    parser.add_argument('--test_interval', type=int, default=5, help="test per dataloader")

    #### model architecture options
    parser.add_argument('--use_lora', type=bool, default=False, help='adopt lora')
    parser.add_argument('--use_Sklar', type=bool, default=False, help='adopt multi-view priors')
    parser.add_argument('--Sklar_type', type=str, default='mvdream', help='the selected model for multi-view priors')
    parser.add_argument('--Sklar_max_iter', type=int, default=1000, help='lambda change in the range of iters')
    parser.add_argument('--Sklar_initial_coef', type=float, default=0.0, help='initial lambda')
    # fine-tune texture for 0.1,train 3D-GS for 0.1-1.0
    parser.add_argument('--Sklar_finial_coef', type=float, default=0.5, help='finial lambda')
    parser.add_argument('--Sklar_coef_method', type=str, default='constant',
                        help='the lambda change method: linear, cosine, exponential, constant')
    parser.add_argument('--Sklar_object', type=str, default='latent',
                        help='way to add the multi-view priors, latent or img ')
    parser.add_argument('--Sklar_method', type=str, default='iteration')
    parser.add_argument('--Sklar_single', type=bool, default=False)

    #### stable diffusion model parameter
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='text-to-image diffusion model for guidance')
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="choose the version")
    parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98], help="time step range")
    parser.add_argument('--scale', type=float, default=7.5, help="CFG")
    parser.add_argument('--t5_iters', type=int, default=2000, help="iters for annealing to t_max = 500")

    #### dataset options
    # train
    parser.add_argument('--angle_front', type=float, default=60.0)
    parser.add_argument('--angle_overhead', type=float, default=-60.0)
    parser.add_argument('--radius_range', type=float, nargs='*', default=[2.0, 2.5])
    parser.add_argument('--elevation_range', type=float, nargs='*', default=[-90, 30])
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[40, 70])
    parser.add_argument('--dir_text', action='store_true', default=True)
    parser.add_argument('--near', type=float, default=0.01)
    parser.add_argument('--far', type=float, default=100)

    # val
    parser.add_argument('--val_radius', type=float, default=3.2)
    parser.add_argument('--val_elevation', type=float, default=-20)
    parser.add_argument('--val_size', type=int, default=4)

    #### 3D-GS options
    parser.add_argument('--sh_degree', type=int, default=0, help="")
    parser.add_argument('--num_pts_init', type=int, default=1000, help="initial number of 3D Gaussian primitives")
    parser.add_argument('--variable_resolution', type=bool, default=True, help='change resolution')
    parser.add_argument('--w', type=int, default=512, help="rendering resolution for 3D-GS")
    parser.add_argument('--h', type=int, default=512, help="rendering resolution for 3D-GS")
    parser.add_argument('--invert_bg_prob', type=float, default=0.5, help="probability for changing background color")
    parser.add_argument('--random_bg_color', type=bool, default=False, help='random background color')

    # 3D-GS optimization parameters
    parser.add_argument('--lr_position_init', type=float, default=0.001, help="learning rate for position, initial")
    parser.add_argument('--lr_position_final', type=float, default=0.00002, help="learning rate for position, finial")
    parser.add_argument('--lr_position_max_steps', type=int, default=1500, help="iter range for change learning rate of position")
    parser.add_argument('--lr_position_delay_mult', type=float, default=0.01)
    parser.add_argument('--lr_feature', type=float, default=0.01, help="learning rate for feature (color)")
    parser.add_argument('--lr_opacity', type=float, default=0.05, help="learning rate for opacity")
    parser.add_argument('--lr_scaling', type=float, default=0.005, help='learning rate for scaling')
    parser.add_argument('--lr_rotation', type=float, default=0.001, help='learning rate for rotation')
    parser.add_argument('--percent_dense', type=float, default=0.01)
    parser.add_argument('--min_opacity', type=float, default=0.01)
    parser.add_argument('--density_start_iter', type=int, default=0, help='start iter for density')
    parser.add_argument('--density_end_iter', type=int, default=1500, help='end iter for density')
    parser.add_argument('--densification_interval', type=int, default=250, help='interval iter for density')
    parser.add_argument('--opacity_reset_interval', type=int, default=200000, help='reset interval iter')
    parser.add_argument('--densify_grad_threshold', type=float, default=0.01)
    parser.add_argument('--mesh_format', type=str, default='obj')

    #### lora parameter
    parser.add_argument('--warm_iters', type=int, default=100)
    parser.add_argument('--K', type=int, default=1, help="learning number for lora, each iter")
    # 3D-GS for 1, dmtet for 10
    parser.add_argument('--K2', type=int, default=1, help="interval iter for learning lora")
    parser.add_argument('--unet_bs', type=int, default=1, help="batch size for lora")
    parser.add_argument('--unet_lr', type=float, default=0.0001, help="learning rate for lora")
    parser.add_argument('--q_cond', type=bool, default=True, help="adopt pose for lora")
    parser.add_argument('--uncond_p', type=float, default=0.1)
    parser.add_argument('--v_pred', type=bool, default=True)

    opt = parser.parse_args()

    if opt.seed is not None:
        seed_everything(opt.seed)
    device = torch.device('cuda:0')
    device_Sklar = torch.device('cuda:1')

    if opt.test:
        opt.workspace = './test_GS/'
        if opt.GS_dmtet:
            if opt.dmtet_finetune:
                dirs = 'stage_3/'
            else:
                dirs = 'stage_2/'
        else:
            dirs = 'stage_1/'
        opt.workspace += dirs
        opt.test_model_path = opt.workspace + opt.test_model_path

        opt.val_size = 12
        model = Renderer(opt, device)

        if len(opt.test_model_path) == 0:
            raise ValueError(f"The test_model_path parameter is empty!")
        else:
            if opt.GS_dmtet:
                model.GS_DMTET.to(device)
                state_dict = torch.load(opt.test_model_path, map_location=device)
                model.GS_DMTET.load_state_dict(state_dict['GS_DMTET'], strict=False)
            else:
                model.initialize(input=opt.test_model_path)

        opt.workspace += os.path.splitext(os.path.basename(opt.test_model_path))[0].replace(" ", "-")

        val_dataset = GSDataset(opt, type='val', H=opt.h, W=opt.w, size=opt.val_size)
        test_dataset = GSDataset(opt, type='test', H=opt.h, W=opt.w, size=200)

        trainer = Trainer(' '.join(sys.argv), opt, model, guidance=None, device=device, workspace=opt.workspace)
        trainer.test(test_dataset, val_dataset, main_test=True)
    else:
        #################### add training parameters into dir name for saving results ####################
        if opt.GS_dmtet:
            opt.t_range = [0.02, 0.50]
            opt.fovy_range = [30, 60]
            opt.variable_resolution = False

            if opt.dmtet_finetune:
                opt.iters = opt.MLP_initial_iters + opt.dmtet_iters
                opt.t5_iters = opt.MLP_initial_iters + opt.dmtet_t5_iters
                opt.scale = 7.5
                opt.workspace = 'stage_3'
                opt.val_interval = 10
                opt.test_interval = 10
            else:
                opt.iters = opt.dmtet_iters
                opt.t5_iters = opt.dmtet_t5_iters
                opt.scale = 100
                opt.workspace = 'stage_2'
                opt.text_short += '_' + str(opt.density_thresh)
                opt.val_interval = 30
                opt.test_interval = 30
        else:
            opt.workspace = 'stage_1'

        if opt.use_Sklar:
            if opt.Sklar_method == 'iteration':
                if opt.Sklar_object == 'latent':
                    opt.workspace += '_iteration_latent/'
                else:
                    opt.workspace += '_iteration_img/'
            else:
                if opt.Sklar_object == 'latent':
                    opt.workspace += '_regular_latent/'
                else:
                    opt.workspace += '_regular_img/'
            if opt.Sklar_single:
                opt.name += 'Single_'
            else:
                opt.name += 'Four_'
            opt.workspace += "Sklar-" + opt.Sklar_type
            if opt.Sklar_coef_method == 'constant':
                opt.workspace += "-lambda-" + str(opt.Sklar_finial_coef) + "-"
            else:
                opt.workspace += ("-" + opt.Sklar_coef_method + "-" + str(opt.Sklar_max_iter) + "-" +
                                  str(opt.Sklar_initial_coef) + "-" + str(opt.Sklar_finial_coef) + "-")
        else:
            opt.workspace += '/'

        if opt.opacity_reset_interval < opt.iters:
            opt.name += ('density-' + str(opt.opacity_reset_interval) + '-' + str(opt.densification_interval) + '-range-' +
                         str(opt.density_start_iter)) + '-' + str(opt.density_end_iter)
        else:
            opt.name += 'no_density' + '-' + str(opt.densification_interval)
        if opt.use_lora:
            opt.workspace += "Lora-"
        if opt.text is not None:
            opt.workspace += "Text-" + str(opt.text_short).replace(" ", "-")
            # opt.workspace += "Text-"+str(opt.text).replace(" ", "-")

        opt.workspace += '-sd_' + str(opt.sd_version)
        opt.workspace += "-scale-" + str(opt.scale)
        opt.workspace += "-iters-" + str(opt.iters)

        if opt.seed is not None:
            opt.workspace += "-seed-" + str(opt.seed)

        if opt.t5_iters != -1:
            opt.workspace += "-t5-" + str(opt.t5_iters)
        if not opt.GS_dmtet:
            opt.workspace += "-num_pt-" + str(opt.num_pts_init)
        if opt.variable_resolution:
            opt.workspace += '-Change-'
        if opt.random_bg_color:
            opt.workspace += '-Random_bg'
        if opt.use_lora:
            opt.workspace += '-Lora-K2-' + str(opt.K2) + '-K-' + str(opt.K)

        #################### Preparation before training ####################

        # define 3D-GS
        model = Renderer(opt, device)
        if opt.GS_dmtet:
            model.GS_DMTET.to(device)

            # texture
            if opt.dmtet_finetune:
                state_dict = torch.load(opt.finetune_path, map_location=device)
                model.GS_DMTET.load_state_dict(state_dict['GS_DMTET'], strict=False)
                # load scaled 3D-GS
                model.initialize(input=opt.dmtet_init_path_scale, scale=state_dict['scale'])
                print('Finished the GS_dmtet_finetune initial!')
            # geometry
            else:
                # load optimized 3D-GS
                model.initialize(input=opt.dmtet_init_path)
                # initialize dmtet
                model.init_tet()
                print('Finished the GS_dmtet initial!')
        else:
            if opt.checkpoint_stage1 is not None:
                model.initialize(input=opt.checkpoint_stage1)
            else:
                # randomly initialize 3D-GS
                model.initialize(num_pts=opt.num_pts_init)

        train_dataset = GSDataset(opt, type='train', H=opt.h, W=opt.w, size=opt.iter_per_loader)
        val_dataset = GSDataset(opt, type='val', H=opt.h, W=opt.w, size=opt.val_size)
        test_dataset = GSDataset(opt, type='test', H=opt.h, W=opt.w, size=100)

        if opt.GS_dmtet:
            optimizer = torch.optim.Adam(model.GS_DMTET.get_params(opt.dmtet_lr, opt.albedo_lr), betas=(0.9, 0.99), eps=1e-15)
        else:
            optimizer = model.gaussians.training_setup(opt)
            model.gaussians.active_sh_degree = model.gaussians.max_sh_degree

        # define guidance model
        if opt.guidance == 'stable-diffusion':
            from guidance.sd import StableDiffusion
            guidance = StableDiffusion(device, opt.sd_version, opt)
        else:
            raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

        Sklar_model = None
        if opt.use_Sklar:
            if opt.Sklar_method == 'iteration':
                from guidance.mvdream_utils_iteration import MVDream
                Sklar_model = MVDream(device_Sklar, opt)

        trainer = Trainer(' '.join(sys.argv), opt, model, guidance, optimizer=optimizer, device=device,
                          val_interval=opt.val_interval, workspace=opt.workspace,
                          use_tensorboardX=opt.use_tensorboardX, lora_scheduler_update_every_iter=True,
                          Sklar_model=Sklar_model, device_Sklar=device_Sklar)
        max_epoch = np.ceil(opt.iters / opt.iter_per_loader).astype(np.int32)
        # start training
        trainer.train(train_dataset, val_dataset, test_dataset, max_epoch)
