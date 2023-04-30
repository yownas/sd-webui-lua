# sd-webui-lua
Generate images with Lua in Stable Diffusion webui.

This is an extension for [Vlad's automatic](https://github.com/vladmandic/automatic/) or [automatic1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) to let you run Lua code to generate images (and more?).

It is also NOT a serious extension (yet). It is mostly just an experiment from my side to see if it was possible to do or not. :)

# Issues

* Gradios Gallery seem to freeze if it get and empty result. "Solved" at the moment by giving it a 1x1 pixel place holder.

* Needs a lot more functions to manipulate/access things. (and a save button/function)

# Lua commands

`sd.empty_latent`: Get an empty latent (not used yet)

`sd.pipeline`: Generate image from string

`sd.process`: Generate image from string or p

`sd.getp`: return an empty p object

`sd.cond`: parse prompt

`sd.negcond`: parse negative prompt (WHY are these different???)

`sd.sample`: Do the thing

`sd.vae`: Run latent through vae

`sd.toimage`: Get an image (maybe) from latent

`sd.save`: Save. (not implemented yet)

`ui.clear`: Clear the Output text

`ui.console`: Print to console

`ui.out`: Add text to Output

`ui.gallery.add`: Add image to Gallery

`ui.gallery.clear`: Clear Gallery


