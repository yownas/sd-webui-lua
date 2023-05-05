# sd-webui-lua
Generate images with Lua in Stable Diffusion webui.

This is an extension for [Vlad's automatic](https://github.com/vladmandic/automatic/) or [automatic1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) to let you run Lua code to generate images (and more?).

# Usage

It way too early to even try writing some sort of guide. You can see all (I hope) functions if you open up the "Lua Extras..." at the bottom, but the best way to get started is probably by looking in the `examples` folder and tinker with those.

![sd-webui-lua](https://user-images.githubusercontent.com/13150150/235615238-a92f6395-d6f7-4e03-8d52-095edeb8aef2.png)

# Issues & todo

* Gradios Gallery seem to freeze if it get and empty result. "Solved" at the moment by giving it a 1x1 pixel place holder. 

* Textarea should be a gr.Code box instead, but there seem to be problems with the css.

* Needs a lot more functions to manipulate/access things. (and a save button/function)

* Some kind of progress-bar/feedback while running would be nice.

* More examples to show how things work.

* More ideas. I'd rather have a PR than a vague Issue. But I'm happy for any input. :)

