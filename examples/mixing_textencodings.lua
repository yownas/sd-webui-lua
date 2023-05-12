-- Mixing two Clip encoded texts - Yownas
prompt1 = [[photograph portrait, magic forest, cute elf picking flowers, sparkles]]

prompt2 = [[cyberpunk city, derelict ruins, monsters, scary dark]]

uc = sd.negcond('nsfw, out of focus, blurry, boring and other things we do not want to see')

-- We want garbage collecting running
collectgarbage(restart)
p = sd.getp()
p.seed = -1
p.width = 640
p.height = 512

ui.status("Encoding text")
c1 = sd.textencode(prompt1)
c2 = sd.textencode(prompt2)
e = torch.new_zeros({1, 77, 768})

-- Copy half of the tensors from c1 and half from c2
ui.status("Mixing")
for x = 0, 76, 1 do
  for y = 0, 767, 2 do
    e[0][x][y] = c1[0][x][y]
  end
  for y = 1, 767, 2 do
    e[0][x][y] = c2[0][x][y]
  end
end

ui.status("Generating image 1/3")
ui.gallery.addc(sd.toimage(sd.vae(sd.sample(p, c1, uc))), "Original prompt 1")
ui.status("Generating image 2/3")
ui.gallery.addc(sd.toimage(sd.vae(sd.sample(p, c2, uc))), "Original prompt 2")
ui.status("Generating image 3/3")
ui.gallery.addc(sd.toimage(sd.vae(sd.sample(p, e, uc))), "Mixed")
