-- Playing "telephone" - Yownas
p = sd.getp()

p.prompt = [[view of a futuristic city at night,
quantum technology, stars, moon, night sky, neon signs,
highly detailed, 4k uhd, sci-fi, intricate detail,
high quality, sharp focus, 2.8f, fractal landscape]]

p.negative_prompt = [[nsfw, illustration, drawing,
painting, out of focus, blurry, distorted, 3d, sketch,
digital art, watermark, signature, hands, giger]]

p.seed = -1
p.width = 640
p.height = 480
p.steps = 10
p.sampler = "UniPC"
p.cfg_scale = 3

for i = 1, 12, 1 do
  ui.status("Generating: " .. i)
  ui.console(p.prompt)
  img = sd.restorefaces(sd.pipeline(p))
  ui.gallery.addc(img, i)
  p.prompt = sd.interrogate.clip(img)
end

return("Seed: " .. p.seed)
