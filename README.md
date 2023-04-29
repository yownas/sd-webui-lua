# sd-webui-lua
Generate images with Lua in Stable Diffusion webui.

This is an extension for [Vlad's automatic](https://github.com/vladmandic/automatic/) or [automatic1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) to let you run Lua code to generate images (and more?).

It is also NOT a serious extension. It is mostly just an experiment from my side to see if it was possible to do or not. Do not use it. :)

# Issues

* Gradios Gallery seem to freeze if it get and empty result. "Solved" at the moment by giving it a 1x1 pixel place holder.

* Needs a lot more hooks into the functions to generate images.

# Lua commands

`ui.console(text)` Print text string console. (Same as print()) 

`ui.gallery.add(image)` Add image to the gallery.

`ui.gallery.clear()` Clear the gallery.

`ui.out(text)` Add text to Output textbox.

`ui.clear()` Clear Output textbox.

`process(prompt)` Process a string, returns an image.

`clip(text)` 

