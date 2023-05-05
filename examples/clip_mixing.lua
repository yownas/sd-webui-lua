-- Mix Clip encoded texts - Yownas
prompt = 'cat'
prompt2 = 'dog'

p = sd.getp()
p.seed = 42
uc = sd.negcond('')

p1 = sd.textencode(prompt)
p2 = sd.textencode(prompt2)

for weight = 0, 1, 0.1
do
  a = torch.lerp(p1, p2, weight)
  c = sd.negcond2cond(sd.clip2negcond(a))
  latent = sd.sample(p, c, uc)
  ui.gallery.addc(sd.toimage(sd.vae(latent)), weight)
end

gif = ui.gallery.getgif(200)
ui.gallery.addc(gif, 'Cat to dog gif')
