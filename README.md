# sd-webui-lua
Generate images with Lua scripts in Stable Diffusion webui.

This is an extension for [Vlad's SD.next](https://github.com/vladmandic/automatic/) and [automatic1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) to let you run Lua code to generate images (and more?).

# Installation

Under `Extensions` go to `Manual install` and paste the link to this repository. Click `Install`.

# Usage

It too early to try writing some sort of proper guide. But all functions should be listed if you open up the "Lua Extras..." at the bottom, but the best way to get started is probably by looking in the `examples` folder and tinker with those. There are also some information in the wiki.

![sd-webui-lua](https://github.com/yownas/sd-webui-lua/assets/13150150/f5bbd9f0-1519-4219-8ff0-a296c4ec1172)

# Why?

This project started as a joke, but quickly turned into a fun way to dig into and learn more about the code behind the webui. It also turned out
to be a fun tool to generate images in weird ways that normally isn't really possible through the webui. So, is this an amazing tool to help you
generate beautiful images? No. But it will probably be fun to play with if you know a little about how Stable Diffusion works, but not enough
to be comfortable to write your own code from scratch.

With this extension you can let the webui do the heavy lifting but still have access to poke at things on a fairly low level.

# Issues & todo

* Better css for the poor Code-box.

* Needs a lot more functions to manipulate/access things. (and a save button/function)

* Split sd.sample() into more parts.

* Maybe import the diffusers-library, but I'm not sure how to use that with ckpt/safetensors.

* More examples to show how things work. (Or actual documentation)

* More ideas. I'd rather have a PR than a vague Issue. But I'm happy for any input. :)

