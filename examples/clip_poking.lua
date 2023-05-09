-- Mess with the Clip encoded text - Yownas
prompt = 'cat'

p = sd.getp()
p.seed = 1234
c= sd.cond(prompt)
uc = sd.negcond('')

latent = sd.sample(p, c, uc)
ui.gallery.addc(sd.toimage(sd.vae(latent)), 'Original')

a = sd.textencode(prompt)

ui.gallery.addc(sd.toimage(a), 'Original text encode')

-- Print the tensor and size to Output
ui.out(a)
ui.out(torch.size(a))

-- Poke the encoded text (Just change one of the 77*768 numbers)
a[0][0][0] = 1.0

ui.gallery.addc(sd.toimage(a), 'Text encode after change')

c = sd.negcond2cond(sd.clip2negcond(a))
latent = sd.sample(p, c, uc)
ui.gallery.addc(sd.toimage(sd.vae(latent)), 'Poked latent')
