-- Mix Clip encoded texts - Yownas

prompt = 'dog'
prompt2 = 'cat'

-- We need to clear the gallery before we start.
-- ui.gallery.getgif() will fail later if there is anthing
-- but images in the gellery.
ui.gallery.clear()

p = sd.getp()
p.seed = 4

-- Get encoded text from both prompts
p1 = sd.textencode(prompt)
p2 = sd.textencode(prompt2)

for weight = 0, 1, 0.1
do
  -- Mix p1 and p2
  c = torch.lerp(p1, p2, weight)
  -- sd.sample wants a conditional and unconditional object
  -- but will also take a string as argument and change it
  -- to what it needs, so we don't have to bother creating
  -- a proper unconditional value for it here.
  latent = sd.sample(p, c, '')
  ui.gallery.addc(sd.toimage(sd.vae(latent)), weight)
end

-- Create a gif from the images in the gallery
gif = ui.gallery.getgif(200)
ui.gallery.addc(gif, 'Dog to cat gif')
