# sd-webui-lua
Generate images with Lua in Stable Diffusion webui.

This is an extension for [Vlad's automatic](https://github.com/vladmandic/automatic/) or [automatic1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) to let you run Lua code to generate images (and more?).

It is also NOT a serious extension (yet). It is mostly just an experiment from my side to see if it was possible to do or not. :)

![sd-webui-lua](https://user-images.githubusercontent.com/13150150/235615238-a92f6395-d6f7-4e03-8d52-095edeb8aef2.png)

# Issues

* Gradios Gallery seem to freeze if it get and empty result. "Solved" at the moment by giving it a 1x1 pixel place holder.

* Textarea should be a gr.Code box instead, but there seem to be problems with the css.

* Needs a lot more functions to manipulate/access things. (and a save button/function)


