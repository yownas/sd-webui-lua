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

-- Change values in the range -1 to 1 -> 0 to 1
tmp = torch.add(latent, 1.0)
tmp = torch.mul(tmp, 0.5)
tmp = torch.clamp(tmp, 0.0, 1.0)
img = sd.toimage(tmp)
ui.gallery.add(img)

vae = sd.vae(latent)
img = sd.toimage(vae)
ui.gallery.add(img)

