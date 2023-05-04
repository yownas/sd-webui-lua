p = sd.getp()
c = sd.cond('bunny')
uc = sd.negcond('banana')
latent = sd.sample(p, c, uc)

-- Generate original images
tmp = torch.clamp(torch.mul(torch.add(latent, 1.0), 0.5), 0.0, 1.0)
ui.gallery.add(sd.toimage(tmp))

ui.gallery.add(sd.toimage(sd.vae(latent)))

-- The latent space we got from the sampler has
-- shape [1][4][64][64], lets put random numbers
-- in the upper left corner.
-- (Yes, this is Lua, but the tensors start at 0.)
for z = 0, 3, 1 do
  for x = 0, 32, 1 do
    for y = 0, 32, 1 do
      -- Poke the tensor
      latent[0][z][x][y] = math.random()
    end
  end
end

-- Generate new images
tmp = torch.clamp(torch.mul(torch.add(latent, 1.0), 0.5), 0.0, 1.0)
ui.gallery.add(sd.toimage(tmp))

ui.gallery.add(sd.toimage(sd.vae(latent)))
