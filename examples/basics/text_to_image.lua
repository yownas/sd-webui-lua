-- text_to_image.lua - Different way to generate images - Yownas

-- Simple, with hardcoded defaults
img = sd.process('a cute cat')
ui.gallery.add(img)

-- Simple, using a Processing object (with caption)
p = sd.getp()
p.prompt = 'a cute puppy'
p.negative_prompt = 'angry bear'
p.steps = 25
img = sd.pipeline(p)
ui.gallery.addc(img, 'Not an angry bear')

-- Do the steps manually, also show latent before it is parsed by vae
p = sd.getp()
c = sd.cond('bunny')
uc = sd.negcond('banana')
latent = sd.sample(p, c, uc)
tmp = torch.clamp(torch.mul(torch.add(latent, 1.0) , 0.5), 0.0, 1.0) -- Convert range -1..1 to 0..1
img = sd.toimage(tmp)
ui.gallery.add(img)
vae = sd.vae(latent)
img = sd.toimage(vae)
ui.gallery.add(img)
