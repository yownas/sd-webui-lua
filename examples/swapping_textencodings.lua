-- Randomly swapping numbers in two Clip encoded texts - Yownas
prompt1 = [[painting concept art of a futuristic city in a future city at night,
highly detailed, concept art,4k uhd, digital painting, octane render, sci-fi,
highly detailed, intricate detail, high quality, photorealistic, 4k uhd render,
sharp focus, hdr, smooth, sharp focus]]

prompt2 = [[portrait of a female steampunk wizard casting a magic spell in a magical forest, fantasy,
intricate, elegant, highly detailed, digital painting, artstation, concept art, matte,
sharp focus, illustration, 4k ultra hd, illustration, natural lighting, octane render,
sparkles, vivid colors, fractal landscape, smiling, rainbow]]

uc = sd.negcond('out of focus, blurry, distorted, watermark, signature')

-- We want garbage collecting running
collectgarbage(restart)
p = sd.getp()
p.seed = -1
p.width = 640
p.height = 512
p.steps = 15
p.sampler = "UniPC"
p.cfg_scale = 4

ui.status("Encoding text")
c1 = sd.textencode(prompt1)
c2 = sd.textencode(prompt2)
ui.status("Generating image 1/4")
ui.gallery.addc(sd.toimage(sd.vae(sd.sample(p, c1, uc))), "Original prompt 1")
ui.status("Generating image 2/4")
ui.gallery.addc(sd.toimage(sd.vae(sd.sample(p, c2, uc))), "Original prompt 2")

-- Swap some random tensors
ui.status("Mixing")
for i = 0, 50000, 1 do
  x = math.random(76)
  y = math.random(767)
  t = torch.t2f(c1[0][x][y]) -- convert into a float
  c1[0][x][y] = c2[0][x][y]
  c2[0][x][y] = t
end

ui.status("Generating image 3/4")
ui.gallery.addc(sd.restorefaces(sd.toimage(sd.vae(sd.sample(p, c1, uc)))), "Mixed 1")
ui.status("Generating image 4/4")
ui.gallery.addc(sd.restorefaces(sd.toimage(sd.vae(sd.sample(p, c2, uc)))), "Mixed 2")

ui.status("Done.")
-- Return the seed to the Output box
return(p.seed)
