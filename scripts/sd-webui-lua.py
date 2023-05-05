from collections import namedtuple
import gradio as gr
import lupa
import numpy as np
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import traceback

from modules import scripts, script_callbacks, devices, ui, shared, processing, sd_samplers, sd_samplers_common, paths
from modules import prompt_parser

import modules.images as images

from modules.shared import opts, cmd_opts, state

from modules.processing import StableDiffusionProcessingTxt2Img, Processed, process_images, fix_seed, decode_first_stage, apply_overlay, apply_color_correction, create_infotext, create_random_tensors

def filter_attribute_access(obj, attr_name, is_setting):
    if isinstance(attr_name, (str)):
        if not attr_name.startswith('_'):
            return attr_name
    raise AttributeError('access denied')

L = lupa.LuaRuntime(register_eval=False, attribute_filter=filter_attribute_access)
G = L.globals()
LUA_output = ''
LUA_gallery = []

def lua_run(lua_code):
    global LUA_output, LUA_gallery
    try:
        result = L.execute(lua_code)
        if result:
            LUA_output += str(result)+'\n'
    except Exception as err:
        traceback.print_exc()
        result = f"ERROR: {err}"
        print(f"LUA {result}")
        raise gr.Error(result)
    # Weird work-around, gr.Gallery seem to freeze the ui if it get an empty reply https://github.com/gradio-app/gradio/issues/3944
    return LUA_output, LUA_gallery if len(LUA_gallery) else [Image.frombytes("L", (1, 1), b'\x00')]

def lua_reset():
    global L, G, LUA_output, LUA_gallery
    L = lupa.LuaRuntime(register_eval=False, attribute_filter=filter_attribute_access)
    G = L.globals()
    LUA_output = ''
    LUA_gallery = []
    # Setup python functions (messy list. Will most likely change)
    G.sd = {
            'empty_latent': sd_lua_empty_latent,
            'pipeline': sd_lua_pipeline,
            'process': sd_lua_process,
            'getp': sd_lua_getp,
            'cond': sd_lua_cond,
            'negcond': sd_lua_negcond,
            'sample': sd_lua_sample,
            'vae': sd_lua_vae,
            'toimage': sd_lua_toimage,
            'makegif': sd_lua_makegif,

            'textencode': sd_lua_textencode,
            'clip2negcond': sd_lua_clip2negcond,
            'negcond2cond': sd_lua_negcond2cond,

        }
    G.ui = {
            'clear': ui_lua_output_clear,
            'console': ui_lua_console,
            'out': ui_lua_output,
            'gallery': {
                'add': ui_lua_gallery_add,
                'addc': ui_lua_gallery_addc,
                'clear': ui_lua_gallery_clear,
                'del': ui_lua_gallery_del,
                'getgif': ui_lua_gallery_getgif
                },
            'image': {
                'save': ui_lua_imagesave,
                }
        }
    G.torch = {
                'add': torch_add,
                'clamp': torch_clamp,
                'lerp': torch_lerp,
                'mul': torch_mul,
                'size': torch_size,
                'new_zeros': torch_new_zeros,
            }
    return LUA_output, LUA_gallery if len(LUA_gallery) else [Image.frombytes("L", (1, 1), b'\x00')]

def lua_refresh():
    global LUA_output, LUA_gallery
    return LUA_output, LUA_gallery if len(LUA_gallery) else [Image.frombytes("L", (1, 1), b'\x00')]

# Functions for Lua

def ui_lua_console(text):
    print(f"Lua: {text}")

def ui_lua_output(text):
    global LUA_output
    LUA_output += str(text)+'\n'

def ui_lua_output_clear():
    global LUA_output
    LUA_output = ''

def ui_lua_gallery_add(image):
    ui_lua_gallery_addc(image, '')

def ui_lua_gallery_addc(image, caption):
    global LUA_gallery
    LUA_gallery.insert(0, (image, caption))

def ui_lua_gallery_getgif(duration):
    global LUA_gallery
    gif = []
    for i in LUA_gallery:
        gif.insert(0, i[0])
    gif_path = sd_lua_list2gif(gif, duration)
    return(gif_path)

def ui_lua_gallery_clear():
    global LUA_gallery
    LUA_gallery = []

def ui_lua_gallery_del(index):
    global LUA_gallery
    # FIXME add code here to match caption
    del LUA_gallery[index-1]

# Empty latent
# IN: width, height
# OUT: latent
def sd_lua_empty_latent (w, h):
    tensor = torch.tensor((), dtype=torch.float32)
    return tensor.new_zeros((w, h))

# IN:
# OUT: p
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
        denoising_strength=0,
        hr_scale=0,
        hr_upscaler=None,
        hr_second_pass_steps=0,
        hr_resize_x=0,
        hr_resize_y=0,
        override_settings=[],
    )
    return(p)

# Conditioning functions
ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])
class ComposableScheduledPromptConditioning:
    def __init__(self, schedules, weight=1.0):
        self.schedules: List[ScheduledPromptConditioning] = schedules
        self.weight: float = weight
class MulticondLearnedConditioning:
    def __init__(self, shape, batch):
        self.shape: tuple = shape  # the shape field is needed to send this object to DDIM/PLMS
        self.batch: List[List[ComposableScheduledPromptConditioning]] = batch

def sd_lua_cond(prompt):
    with devices.autocast():
        cond = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, [prompt], 1000)
    return(cond)
def sd_lua_negcond(prompt):
    with devices.autocast():
        cond = prompt_parser.get_learned_conditioning(shared.sd_model, [prompt], 1000)
    return(cond)

def sd_lua_textencode(prompt): # Prompt to learned conditioning
    with devices.autocast():
        conds = shared.sd_model.get_learned_conditioning([prompt])
    return(conds)

def sd_lua_clip2negcond(conds): # Probably need a better name than clip
    res = []
    cond_schedule = []

    with devices.autocast():
        for cond in conds:
            cond_schedule.append(ScheduledPromptConditioning(1000, cond)) # Hardcoded steps, hopefully not breaking too many things
    res.append(cond_schedule)
    return(res)

def sd_lua_negcond2cond(negcond):
    res = []
    cond_schedule = []
    with devices.autocast():
        for spcs in negcond:
            cond_schedule.append([ComposableScheduledPromptConditioning(spcs, 1.0)])

    # [[<modules.prompt_parser.ComposableScheduledPromptConditioning object at 0x7f5031e161a0>]]
    return MulticondLearnedConditioning(shape=(1,), batch=cond_schedule)

# IN: p, c, uc
# OUT: latent
def sd_lua_sample(p, c, uc):
    fix_seed(p)
    with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
        samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=[p.seed], subseeds=[p.subseed], subseed_strength=p.subseed_strength, prompts=[p.prompt])
    return(samples_ddim)

# IN: latent
# OUT: latent
def sd_lua_vae(samples_ddim):
    x_samples_ddim = [decode_first_stage(shared.sd_model, samples_ddim.to(dtype=devices.dtype_vae))[0].cpu()]
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
    return(x_samples_ddim)

# IN: latent
# OUT: image (maybe)
def sd_lua_toimage(latent):
    for i, x_sample in enumerate(latent):
        x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
        x_sample = x_sample.astype(np.uint8)

        image = Image.fromarray(x_sample)

    devices.torch_gc()

    return image

def sd_lua_list2gif(gif, duration):
    path_to_save = os.path.join(opts.outdir_extras_samples, 'lua')
    if not os.path.exists(path_to_save):
        try:
            os.makedirs(path_to_save, exist_ok=True)
            print('LUA: Creating folder:', path_to_save)
        except:
            pass
    name = images.get_next_sequence_number(path_to_save, '')
    path_to_save = os.path.join(path_to_save, f"{name}.gif")
    gif[0].save(path_to_save, save_all=True, append_images=gif[1:], optimize=False, duration=duration, loop=0)
    return(path_to_save)
    
# IN: Lua-table with images
# OUT: string, name of gif
def sd_lua_makegif(table, duration):
    gif = [x for x in table.values()]
    gif_path = sd_lua_list2gif(gif, duration)
    return(gif_path)

# IN: image
# OUT: string (path to image)
def ui_lua_imagesave(image, name):
    path_to_save = os.path.join(opts.outdir_extras_samples, 'lua')
    if not os.path.exists(path_to_save):
        try:
            os.makedirs(path_to_save, exist_ok=True)
            print('LUA: Creating folder:', path_to_save)
        except:
            pass
    path_to_save = os.path.join(path_to_save, name)
    # FIXME check so it doesn't overwrite images?
    image.save(path_to_save)
    return(path_to_save)

# IN: p
# OUT: image
def sd_lua_pipeline(p):
    devices.torch_gc()

    fix_seed(p)

    seed = p.seed
    subseed = p.subseed

    comments = {}

    # FIXME remove? ignoring infotext will make things simpler
    p.all_prompts = [p.prompt]
    p.all_negative_prompts = [p.negative_prompt]
    p.all_seeds = [int(seed)]
    p.all_subseeds = [int(subseed)]

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(p, [p.prompt], [p.seed], [p.subseed], comments, iteration, position_in_batch)

    infotexts = []
    output_images = []

    #with torch.no_grad(), p.sd_model.ema_scope():
    with torch.no_grad():
        prompts = [p.prompt]
        negative_prompts = [p.negative_prompt]
        seeds = [p.seed]
        subseeds = [p.subseed]

        c = sd_lua_cond(p.prompt)
        uc = sd_lua_negcond(p.negative_prompt)

        # Sample
        samples_ddim = sd_lua_sample(p, c, uc)

        x_samples_ddim = sd_lua_vae(samples_ddim)

        del samples_ddim

        if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
            lowvram.send_everything_to_cpu()

        devices.torch_gc()

        for i, x_sample in enumerate(x_samples_ddim):
            x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
            x_sample = x_sample.astype(np.uint8)


            image = Image.fromarray(x_sample)

            if opts.samples_save and not p.do_not_save_samples:
                images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(0, i), p=p)

            text = infotext(0, i)
            infotexts.append(text)
            if opts.enable_pnginfo:
                image.info["parameters"] = text
            output_images.append(image)

        del x_samples_ddim

    devices.torch_gc()

    return output_images[0]

############################################################################3

# IN: p or string
# OUT: image
def sd_lua_process(prompt):
    if isinstance(prompt, str):
        p = sd_lua_getp()
        p.prompt = prompt
    else:
        p = prompt

    processed = process_images(p)
    p.close()
    return processed.images[0]

### Torch functions

def torch_lerp(v1, v2, weight):
    return torch.lerp(v1, v2, weight)

def torch_add(v1, v2):
    return torch.add(v1, v2)

def torch_mul(v1, v2):
    return torch.mul(v1, v2)

def torch_clamp(v1, min, max):
    return torch.clamp(v1, min=0.0, max=1.0)

def torch_size(v1):
    return v1.size()

def torch_new_zeros(size):
    tensor = torch.tensor((), dtype=torch.float32)
    # Lua will return a "table", which for python looks like a dict, so we need to
    # pick  out the values (skipping the keys) and make it into a tuple
    return tensor.new_zeros(tuple(size.values()), device=torch.device('cuda'))

def torch_copy(v):
    return copy(v)

def add_tab():
    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Row():
            with gr.Column(scale=1):
                # Would use this if the css wasn't broken
                #lua_code = gr.Code(label="Lua", language=None, show_label=False, lines=30, placeholder="(Lua code)")
                lua_code = gr.Textbox(label="Lua", show_label=False, lines=30, placeholder="(Lua code)")
                with gr.Row():
                    run = gr.Button('Run', variant='primary')
                    reset = gr.Button('Reset')
                    refresh = gr.Button('Refresh')
            with gr.Column(scale=1):
                with gr.Row():
                    gallery = gr.Gallery(label="Gallery").style(preview=True, grid=4)
                with gr.Row():
                    results = gr.Textbox(label="Output", show_label=True, lines=10)

        run.click(lua_run, show_progress=True, inputs=[lua_code], outputs=[results, gallery])
        reset.click(lua_reset, show_progress=False, inputs=[], outputs=[results, gallery])
        refresh.click(lua_refresh, show_progress=False, inputs=[], outputs=[results, gallery])
        with gr.Row():
            with gr.Accordion(label='Lua Extras...', open=False):
                gr.HTML(
                """
sd-webui-lua links: <a grep=http://github.com/yownas/sd-webui-lua/>Github</a>

<p>
<h2>Functions:</h2>
<p>
<b>ui.out(string):</b><br>
Write string to the Output box.
</p>
<p>
<b>ui.clear():</b><br>
Clear Output box.
</p>
<p>
<b>sd.empty_latent():</b><br>
Get a latent filled with zeroes. (Not used at the moment)
</p>
<p>
<b>sd.pipeline(p):</b><br>
Deconstructed pipeline from the webui.Generate picture from processing object.
</p>
<p>
<b>sd.process(string):</b><br>
Webui pipeline, generate image from a prompt-string or processing object.
</p>
<p>
<b>sd.getp():</b><br>
</p>
Returns a default processing object (see below).
<p>
<b> sd.cond(string):</b><br>
Run prompt string through clip.
</p>
<p>
<b>sd.negcond(string)</b><br>
Run negative prompt string through clip. (These are unfortunately slightly different at the momemt)
</p>
<p>
<b>sd.sample(latent):</b><br>
Turn noise into something that can get turned into an image.
</p>
<p>
<b>sd.vae(latent):</b><br>
Variational auto-envoder.
</p>
<p>
<b>sd.toimage(latent):</b><br>
Last step to get an image after the vae
</p>
<p>
<b>ui.clear():</b><br>
Clear everything.
</p>
<p>
<b>ui.console(string):</b><br>
Print to console. 
</p>
<p>
<b>ui.out(string):</b><br>
Print to Output box.
</p>
<p>
<b>ui.gallery.add(image):</b><br>
Add image to Gallery
</p>
<p>
<b>ui.gallery.addc(image, string):</b><br>
Add image with caption to Gallery.
</p>
<p>
<b>ui.gallery.clear():</b><br>
Clear the gallery.
</p>
<p>
<b>ui.gallery.del(int):</b><br>
Delete image from galler. (Starts at 1 since this is Lua.)
</p>
<p>
<b>ui.gallery.getgif(duration):</b><br>
Get a gif from the images in the gallery. Show each image for "duration" ms.
</p>
<p>
<b>ui.image.save(image, name):</b><br>
Same image.
</p>
<p>
<b>torch_clamp(v1, min, max):</b><br>
Clamp vector v1 between min and max.
</p>
<p>
<b>torch.lerp(v1, v2, weight):</b><br>
Linear interpolation of v1 and v2, by weight. v1 + weight * (v2 - v1)
</p>
<p>
<b>torch.add(v1, v2):</b><br>
Add v2 (vector or float) to v1.
</p>
<p>
<b>torch.mul(v1, v2):</b><br>
Multiply v2 (vector or float) with v1.
</p>
<p>
<b>torch.size(v1):</b><br>
Return the size of vector v1
</p>
<p>
<b>torch.new_zeros(size):</b><br>
Take a Lua table, size, and create a zero-filled tensor.
</p>

<p>
<h2>Default Processing-object:</h2><br>
<pre>
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
 denoising_strength=0,
 hr_scale=0,
 hr_upscaler=None,
 hr_second_pass_steps=0,
 hr_resize_x=0,
 hr_resize_y=0,
 override_settings=[],
)
</pre>
</p>
                """)


    return [(tab, "Lua", "lua")]

x,y = lua_reset()

script_callbacks.on_ui_tabs(add_tab)
