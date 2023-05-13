-- Mixing two Clip encoded texts - Yownas
prompt1 = [[painting concept art of a futuristic city in a future city at night,
highly detailed, concept art,4k uhd, digital painting, octane render, sci-fi,
highly detailed, intricate detail, high quality, photorealistic, 4k uhd render,
sharp focus, hdr, smooth, sharp focus]]

prompt2 = [[portrait of a male wizard casting a magic spell in a dark forest, fantasy,
intricate, elegant, highly detailed, digital painting, artstation, concept art, matte,
sharp focus, illustration, 4k ultra hd, illustration, natural lighting, octane render]]

uc = sd.negcond('out of focus, blurry, distorted, watermark, signature')

-- We want garbage collecting running
collectgarbage(restart)
p = sd.getp()
p.seed = -1
p.width = 640
p.height = 512
p.steps = 15
p.sampler = "UniPC"
p.cfg = 4

ui.status("Encoding text")
c1 = sd.textencode(prompt1)
c2 = sd.textencode(prompt2)

-- Get two empty tensors with the same size as a text-encode
e1 = torch.new_zeros({1, 77, 768})
e2 = torch.new_zeros({1, 77, 768})

-- Copy half of c1 & c2 into e1 & e2
ui.status("Mixing")
for x = 0, 76, 1 do
  for y = 0, 767, 2 do
    -- even numbers; c1 -> e1, c2 -> e2
    e1[0][x][y] = c1[0][x][y]
    e2[0][x][y] = c2[0][x][y]
  end
  for y = 1, 767, 2 do
    -- odd numbers; c2 -> e1, c1 -> e2
    e1[0][x][y] = c2[0][x][y]
    e2[0][x][y] = c1[0][x][y]
  end
end

ui.status("Generating image 1/4")
ui.gallery.addc(sd.toimage(sd.vae(sd.sample(p, c1, uc))), "Original prompt 1")
ui.status("Generating image 2/4")
ui.gallery.addc(sd.toimage(sd.vae(sd.sample(p, c2, uc))), "Original prompt 2")
ui.status("Generating image 3/4")
ui.gallery.addc(sd.toimage(sd.vae(sd.sample(p, e1, uc))), "Mixed 1")
ui.status("Generating image 4/4")
ui.gallery.addc(sd.toimage(sd.vae(sd.sample(p, e2, uc))), "Mixed 2")

-- Return the seed to the Output box
return(p.seed)
