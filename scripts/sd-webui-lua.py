import lupa
from modules import scripts, script_callbacks, devices, ui, shared
import gradio as gr

from modules.processing import StableDiffusionProcessingTxt2Img, Processed, process_images, fix_seed

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
    if result:
        LUA_output = result
    return LUA_output, LUA_gallery

def lua_reset(lua_code):
    global L, G, LUA_output, LUA_gallery
    L = lupa.LuaRuntime(register_eval=False, attribute_filter=filter_attribute_access)
    G = L.globals()
    LUA_output = ''
    LUA_gallery = []
    # Setup python functions
    G.console = sd_lua_console
    G.process = sd_lua_process
    G.gallery_add = sd_lua_gallery_add
    G.gallery_reset = sd_lua_gallery_reset
    return LUA_output, LUA_gallery

# Functions for Lua

def sd_lua_console(text):
    print(f"Lua: {text}")

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

def sd_lua_gallery_add(image):
    global LUA_gallery
    LUA_gallery.append(image)

def sd_lua_gallery_reset():
    global LUA_gallery
    LUA_gallery = []

lua_reset('')

def add_tab():
    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Row():
            lua_code = gr.Textbox(label="Lua", show_label=False, lines=20, placeholder="(Lua code)").style(container=False)
        with gr.Row():
            run = gr.Button('Run', variant='primary')
            reset = gr.Button('Reset')
        with gr.Row():
            results = gr.Textbox(label="Output", show_label=True, lines=10)
            gallery = gr.Gallery(label="Gallery").style(preview=True)

        run.click(lua_run, inputs=[lua_code], outputs=[results, gallery])
        reset.click(lua_reset, inputs=[lua_code], outputs=[results, gallery])

    return [(tab, "Lua", "lua")]

script_callbacks.on_ui_tabs(add_tab)
