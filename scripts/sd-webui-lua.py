import gradio as gr
import lupa
import numpy as np
from PIL import Image
import torch

from modules import scripts, script_callbacks, devices, ui, shared, processing
from modules import prompt_parser
import modules.images as images

from modules.shared import opts, cmd_opts, state

from modules.processing import StableDiffusionProcessingTxt2Img, Processed, process_images, fix_seed, decode_first_stage, apply_overlay, apply_color_correction, create_infotext

def filter_attribute_access(obj, attr_name, is_setting):
    if isinstance(attr_name, unicode):
        if not attr_name.startswith('_'):
            return attr_name
    raise AttributeError('access denied')

L = lupa.LuaRuntime(register_eval=False, attribute_filter=filter_attribute_access)
G = L.globals()
LUA_output = ''
LUA_gallery = []

def lua_run(lua_code):
    global LUA_output, LUA_gallery
    result = L.execute(lua_code)
#FIXME
#    try:
#        result = L.execute(lua_code)
#    except Exception as err:
#        result = f"ERROR: {err}"
#        print(f"LUA {result}")
    if result:
        LUA_output += str(result)+'\n'
    # Weird work-around, gr.Gallery seem to freeze the ui if it get an empty reply
    return LUA_output, LUA_gallery if len(LUA_gallery) else [Image.frombytes("L", (1, 1), b'\x00')]

def lua_reset():
    global L, G, LUA_output, LUA_gallery
    L = lupa.LuaRuntime(register_eval=False, attribute_filter=filter_attribute_access)
    G = L.globals()
    LUA_output = ''
    LUA_gallery = []
    # Setup python functions (there has to be a better way to do this)
    G.sd = {
            'clip': sd_lua_clip,
            'empty_latent': sd_lua_empty_latent,
            'ksampler': sd_lua_ksampler,
            'pipeline': sd_lua_pipeline,
            'process': sd_lua_process
        }
    G.ui = {
            'clear': sd_lua_output_clear,
            'console': sd_lua_console,
            'out': sd_lua_output,
            'gallery': {
                'add': sd_lua_gallery_add,
                'clear': sd_lua_gallery_clear,
                }
        }
#            'gallery_add': sd_lua_gallery_add,
#            'gallery_clear': sd_lua_gallery_clear,
    return LUA_output, LUA_gallery if len(LUA_gallery) else [Image.frombytes("L", (1, 1), b'\x00')]

def lua_refresh():
    global LUA_output, LUA_gallery
    return LUA_output, LUA_gallery if len(LUA_gallery) else [Image.frombytes("L", (1, 1), b'\x00')]

# Functions for Lua

def sd_lua_console(text):
    print(f"Lua: {text}")

def sd_lua_output(text):
    global LUA_output
    LUA_output += str(text)+'\n'

def sd_lua_output_clear():
    global LUA_output
    LUA_output = ''

def sd_lua_gallery_add(image):
    global LUA_gallery
    #LUA_gallery.append(image)
    LUA_gallery.insert(0, image)

def sd_lua_gallery_clear():
    global LUA_gallery
    LUA_gallery = []

# Comfy-inspired functions

# CLIP
# IN: string
# OUT: cond
def sd_lua_clip (prompt):
    with devices.autocast():
        cond = prompt_parser.get_learned_conditioning(shared.sd_model, prompt, 1) # steps hardcoded, some auto1111 tricks won't work
    return cond
        
# Empty latent
# IN: width, height
# OUT: latent
def sd_lua_empty_latent (w, h):
    tensor = torch.tensor((), dtype=torch.float32)
    return tensor.new_zeros((w, h))

# KSampler
# IN: (model), positive conditioning, negative conditioning, latent
# OUT: latent
def sd_lua_ksampler (cond, ncond, latent):
    seed = 1
    subseed = 1
    subseed_strength = 0

    image_cond = latent.new_zeros(latent.shape[0], 5, 1, 1)

    # check img2imgalt

    #x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
    noise = devices.randn(seed, latent.shape)

    #TODO create p?
#    rand_noise = processing.create_random_tensors(latent.shape[1:], seeds=seed, subseeds=subseed, subseed_strength=subseed_strength, seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w, p=p)
#
#    p.sample = sample_extra
#
#
#    samples = self.sampler.sample_img2img(self, latent, noise, cond, ncond, image_cond)

#            with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
#690                samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength, prompts=prompts)


# VAE
# IN: latent, (vae)
# OUT: image
#def sd_lua_vae (prompt):


# REMOVE THIS 
def OLDsd_lua_pipeline(prompt):
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=[],
        negative_prompt='',
        seed=-1,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        seed_enable_extras=True,
        sampler_name='Euler a',
        batch_size=1,
        n_iter=1,
        steps=20,
        cfg_scale=7,
        width=512,
        height=512,
        restore_faces=False,
        tiling=False,
        enable_hr=False,
        denoising_strength=None,
        hr_scale=0,
        hr_upscaler=None,
        hr_second_pass_steps=0,
        hr_resize_x=0,
        hr_resize_y=0,
        override_settings=[],
    )


    if type(p.prompt) == list:
        assert len(p.prompt) > 0
    else:
        assert p.prompt is not None

    devices.torch_gc()

    fix_seed(p)
    seed = p.seed
    subseed = p.subseed

    comments = {}

    if type(p.prompt) == list:
        p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
    else:
        p.all_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

    if type(p.negative_prompt) == list:
        p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in p.negative_prompt]
    else:
        p.all_negative_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]

    if type(seed) == list:
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    if type(subseed) == list:
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

    infotexts = []
    output_images = []

    cached_uc = [None, None]
    cached_c = [None, None]

    def get_conds_with_caching(function, required_prompts, steps, cache):
        """
        Returns the result of calling function(shared.sd_model, required_prompts, steps)
        using a cache to store the result if the same arguments have been used before.
        cache is an array containing two elements. The first element is a tuple
        representing the previously used arguments, or None if no arguments
        have been used before. The second element is where the previously
        computed result is stored.
        """

        if cache[0] is not None and (required_prompts, steps) == cache[0]:
            return cache[1]

        with devices.autocast():
            cache[1] = function(shared.sd_model, required_prompts, steps)

        cache[0] = (required_prompts, steps)
        return cache[1]

    #with torch.no_grad(), p.sd_model.ema_scope():
    with torch.no_grad():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

        if state.job_count == -1:
            state.job_count = p.n_iter

        extra_network_data = None
        for n in range(p.n_iter):
            p.iteration = n

            if state.skipped:
                state.skipped = False

            if state.interrupted:
                break

            prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            if p.scripts is not None:
                p.scripts.before_process_batch(p, batch_number=n, prompts=prompts, seeds=seeds, subseeds=subseeds)

            if len(prompts) == 0:
                break

            if p.scripts is not None:
                p.scripts.process_batch(p, batch_number=n, prompts=prompts, seeds=seeds, subseeds=subseeds)

            step_multiplier = 1
            if not shared.opts.dont_fix_second_order_samplers_schedule:
                try:
                    step_multiplier = 2 if sd_samplers.all_samplers_map.get(p.sampler_name).aliases[0] in ['k_dpmpp_2s_a', 'k_dpmpp_2s_a_ka', 'k_dpmpp_sde', 'k_dpmpp_sde_ka', 'k_dpm_2', 'k_dpm_2_a', 'k_heun'] else 1
                except:
                    pass
            uc = get_conds_with_caching(prompt_parser.get_learned_conditioning, negative_prompts, p.steps * step_multiplier, cached_uc)
            c = get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, p.steps * step_multiplier, cached_c)

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
                samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength, prompts=prompts)

            x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(dtype=devices.dtype_vae))[0].cpu() for i in range(samples_ddim.size(0))]
            try:
                for x in x_samples_ddim:
                    devices.test_for_nans(x, "vae")
            except devices.NansException as e:
                if not shared.cmd_opts.no_half and not shared.cmd_opts.no_half_vae and shared.cmd_opts.rollback_vae:
                    print('\nA tensor with all NaNs was produced in VAE, try converting to bf16.')
                    devices.dtype_vae = torch.bfloat16
                    vae_file, vae_source = sd_vae.resolve_vae(p.sd_model.sd_model_checkpoint)
                    sd_vae.load_vae(p.sd_model, vae_file, vae_source)
                    x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(dtype=devices.dtype_vae))[0].cpu() for i in range(samples_ddim.size(0))]
                    for x in x_samples_ddim:
                        devices.test_for_nans(x, "vae")
                else:
                    raise e

            x_samples_ddim = torch.stack(x_samples_ddim).float()
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            del samples_ddim

            if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
                lowvram.send_everything_to_cpu()

            devices.torch_gc()

            if p.scripts is not None:
                p.scripts.postprocess_batch(p, x_samples_ddim, batch_number=n)

            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                if p.restore_faces:
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_face_restoration:
                        images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-face-restoration")

                    devices.torch_gc()

                    x_sample = modules.face_restoration.restore_faces(x_sample)
                    devices.torch_gc()

                image = Image.fromarray(x_sample)

                if p.scripts is not None:
                    pp = scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image(p, pp)
                    image = pp.image

                if p.color_corrections is not None and i < len(p.color_corrections):
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_color_correction:
                        image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
                        images.save_image(image_without_cc, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-color-correction")
                    image = apply_color_correction(p.color_corrections[i], image)

                image = apply_overlay(image, p.paste_to, i, p.overlay_images)

                if opts.samples_save and not p.do_not_save_samples:
                    images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p)

                text = infotext(n, i)
                infotexts.append(text)
                if opts.enable_pnginfo:
                    image.info["parameters"] = text
                output_images.append(image)

                if hasattr(p, 'mask_for_overlay') and p.mask_for_overlay and any([opts.save_mask, opts.save_mask_composite, opts.return_mask, opts.return_mask_composite]):
                    image_mask = p.mask_for_overlay.convert('RGB')
                    image_mask_composite = Image.composite(image.convert('RGBA').convert('RGBa'), Image.new('RGBa', image.size), images.resize_image(2, p.mask_for_overlay, image.width, image.height).convert('L')).convert('RGBA')

                    if opts.save_mask:
                        images.save_image(image_mask, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask")

                    if opts.save_mask_composite:
                        images.save_image(image_mask_composite, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask-composite")

                    if opts.return_mask:
                        output_images.append(image_mask)

                    if opts.return_mask_composite:
                        output_images.append(image_mask_composite)

            del x_samples_ddim

            devices.torch_gc()

            state.nextjob()

        p.color_corrections = None

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, p.batch_size)

            if opts.return_grid:
                text = infotext()
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                output_images.insert(0, grid)
                index_of_first_image = 1

            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], opts.grid_format, info=infotext(), short_filename=not opts.grid_extended_filename, p=p, grid=True)

    if not p.disable_extra_networks and extra_network_data:
        extra_networks.deactivate(p, extra_network_data)

    devices.torch_gc()

    res = Processed(p, output_images, p.all_seeds[0], infotext(), comments="".join(["\n\n" + x for x in comments]), subseed=p.all_subseeds[0], index_of_first_image=index_of_first_image, infotexts=infotexts)

    return res.images[0]
########################################################################### REMOVE

def sd_lua_getp():
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
        prompt='',
        styles=[],
        negative_prompt='',
        seed=-1,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        seed_enable_extras=True,
        sampler_name='Euler a',
        batch_size=1,
        n_iter=1,
        steps=20,
        cfg_scale=7,
        width=512,
        height=512,
        restore_faces=False,
        tiling=False,
        enable_hr=False,
        denoising_strength=None,
        hr_scale=0,
        hr_upscaler=None,
        hr_second_pass_steps=0,
        hr_resize_x=0,
        hr_resize_y=0,
        override_settings=[],
    )
    return(p)

def sd_lua_pipeline(prompt):
    devices.torch_gc()

    p = sd_lua_getp()

    p.prompt = prompt
    fix_seed(p)

    #FIXME make simpler by just allowing strings?
    if type(p.prompt) == list:
        assert len(p.prompt) > 0
    else:
        assert p.prompt is not None

    seed = p.seed
    subseed = p.subseed

    comments = {}

    # FIXME remove lists. skip styles?
    if type(p.prompt) == list:
        p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
    else:
        p.all_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

    # FIXME remove lists. skip styles?
    if type(p.negative_prompt) == list:
        p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in p.negative_prompt]
    else:
        p.all_negative_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]

    # FIXME remove lists.
    if type(seed) == list:
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    # FIXME remove lists.
    if type(subseed) == list:
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

    infotexts = []
    output_images = []

    #with torch.no_grad(), p.sd_model.ema_scope():
    with torch.no_grad():

        #FIXME not needed?
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

        #FIXME not needed? just one image
        if state.job_count == -1:
            state.job_count = p.n_iter

        extra_network_data = None

        #FIXME just have one image?
        for n in range(p.n_iter):
            p.iteration = n

            #FIXME skip button?
            if state.skipped:
                state.skipped = False

            if state.interrupted:
                break

            prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            if len(prompts) == 0:
                break

            step_multiplier = 1
            if not shared.opts.dont_fix_second_order_samplers_schedule:
                try:
                    step_multiplier = 2 if sd_samplers.all_samplers_map.get(p.sampler_name).aliases[0] in ['k_dpmpp_2s_a', 'k_dpmpp_2s_a_ka', 'k_dpmpp_sde', 'k_dpmpp_sde_ka', 'k_dpm_2', 'k_dpm_2_a', 'k_heun'] else 1
                except:
                    pass

            #FIXME prompts? just a string, if one image?
            with devices.autocast():
                c = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, prompts, p.steps * step_multiplier)
                uc = prompt_parser.get_learned_conditioning(shared.sd_model, negative_prompts, p.steps * step_multiplier)

            #FIXME add function to convert multicond to cond, so we can use same function for c and uc

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            # Diffuse
            with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
                samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength, prompts=prompts)

            x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(dtype=devices.dtype_vae))[0].cpu() for i in range(samples_ddim.size(0))]
            try:
                for x in x_samples_ddim:
                    devices.test_for_nans(x, "vae")
            except devices.NansException as e:
                if not shared.cmd_opts.no_half and not shared.cmd_opts.no_half_vae and shared.cmd_opts.rollback_vae:
                    print('\nA tensor with all NaNs was produced in VAE, try converting to bf16.')
                    devices.dtype_vae = torch.bfloat16
                    vae_file, vae_source = sd_vae.resolve_vae(p.sd_model.sd_model_checkpoint)
                    sd_vae.load_vae(p.sd_model, vae_file, vae_source)
                    x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(dtype=devices.dtype_vae))[0].cpu() for i in range(samples_ddim.size(0))]
                    for x in x_samples_ddim:
                        devices.test_for_nans(x, "vae")
                else:
                    raise e

            x_samples_ddim = torch.stack(x_samples_ddim).float()
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            del samples_ddim

            if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
                lowvram.send_everything_to_cpu()

            devices.torch_gc()

            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                if p.restore_faces:
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_face_restoration:
                        images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-face-restoration")

                    devices.torch_gc()

                    x_sample = modules.face_restoration.restore_faces(x_sample)
                    devices.torch_gc()

                image = Image.fromarray(x_sample)

                if p.color_corrections is not None and i < len(p.color_corrections):
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_color_correction:
                        image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
                        images.save_image(image_without_cc, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-color-correction")
                    image = apply_color_correction(p.color_corrections[i], image)

                #FIXME overlays?
                image = apply_overlay(image, p.paste_to, i, p.overlay_images)

                if opts.samples_save and not p.do_not_save_samples:
                    images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p)

                text = infotext(n, i)
                infotexts.append(text)
                if opts.enable_pnginfo:
                    image.info["parameters"] = text
                output_images.append(image)

                if hasattr(p, 'mask_for_overlay') and p.mask_for_overlay and any([opts.save_mask, opts.save_mask_composite, opts.return_mask, opts.return_mask_composite]):
                    image_mask = p.mask_for_overlay.convert('RGB')
                    image_mask_composite = Image.composite(image.convert('RGBA').convert('RGBa'), Image.new('RGBa', image.size), images.resize_image(2, p.mask_for_overlay, image.width, image.height).convert('L')).convert('RGBA')

                    if opts.save_mask:
                        images.save_image(image_mask, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask")

                    if opts.save_mask_composite:
                        images.save_image(image_mask_composite, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask-composite")

                    if opts.return_mask:
                        output_images.append(image_mask)

                    if opts.return_mask_composite:
                        output_images.append(image_mask_composite)

            del x_samples_ddim

            devices.torch_gc()

            state.nextjob()

        p.color_corrections = None

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, p.batch_size)

            if opts.return_grid:
                text = infotext()
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                output_images.insert(0, grid)
                index_of_first_image = 1

            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], opts.grid_format, info=infotext(), short_filename=not opts.grid_extended_filename, p=p, grid=True)

    if not p.disable_extra_networks and extra_network_data:
        extra_networks.deactivate(p, extra_network_data)

    devices.torch_gc()

    res = Processed(p, output_images, p.all_seeds[0], infotext(), comments="".join(["\n\n" + x for x in comments]), subseed=p.all_subseeds[0], index_of_first_image=index_of_first_image, infotexts=infotexts)

    return res.images[0]


############################################################################3


def sd_lua_process(prompt):
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=[],
        negative_prompt='',
        seed=-1,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        seed_enable_extras=True,
        sampler_name='Euler a',
        batch_size=1,
        n_iter=1,
        steps=20,
        cfg_scale=7,
        width=512,
        height=512,
        restore_faces=False,
        tiling=False,
        enable_hr=False,
        denoising_strength=None,
        hr_scale=0,
        hr_upscaler=None,
        hr_second_pass_steps=0,
        hr_resize_x=0,
        hr_resize_y=0,
        override_settings=[],
    )

    processed = process_images(p)

    p.close()

    return processed.images[0]

def add_tab():
    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Row():
            with gr.Column(scale=1):
                lua_code = gr.Textbox(label="Lua", show_label=False, lines=30, placeholder="(Lua code)").style(container=False)
                with gr.Row():
                    run = gr.Button('Run', variant='primary')
                    reset = gr.Button('Reset')
                    refresh = gr.Button('Refresh')
            with gr.Column(scale=1):
                with gr.Row():
                    gallery = gr.Gallery(label="Gallery").style(preview=True, grid=4)
                with gr.Row():
                    results = gr.Textbox(label="Output", show_label=True, lines=10)

        run.click(lua_run, show_progress=False, inputs=[lua_code], outputs=[results, gallery])
        reset.click(lua_reset, show_progress=False, inputs=[], outputs=[results, gallery])
        refresh.click(lua_refresh, show_progress=False, inputs=[], outputs=[results, gallery])

    return [(tab, "Lua", "lua")]

x,y = lua_reset()

script_callbacks.on_ui_tabs(add_tab)
