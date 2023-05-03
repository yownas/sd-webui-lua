-- text_to_image.lua - Different way to generate images - Yownas

-- Simple, with hardcoded defaults
img = sd.process('a cute cat')
ui.gallery.add(img)

-- Simple, using a Processing object
p = sd.getp()
p.prompt = 'a cute puppy'
p.negative_prompt = 'angry bear'
img = sd.pipeline(p)
ui.gallery.addc(img, 'Not an angry bear')

-- Do the steps manually, also show latent before vae in gallery
p = sd.getp()
c = sd.cond('bunny')
uc = sd.negcond('banana')
latent = sd.sample(p, c, uc)

img = sd.toimage(latent)
ui.gallery.add(img)

vae = sd.vae(latent)
img = sd.toimage(vae)
ui.gallery.add(img)

