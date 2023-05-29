-- Seed travel, making a gif using the images in the Gallery - Yownas
-- Get a Processing-object
p = sd.getp()

p.prompt = 'a cute cat'
p.seed = 42
p.subseed = 1337

-- Change subseed_strength from 0 to 1
for s = 0,1,0.1
do
  p.subseed_strength = s
  img = sd.process(p)
  ui.gallery.addc(img, 'Seed strength: ' .. tostring(s))
end

-- Get a gif from the images in the gallery, show each image 100 ms
gif = ui.gallery.getgif(100)
-- Remove this comment to clear the gallery before adding the gif
-- ui.gallery.clear()
ui.gallery.addc(gif, 'A cute cat gif')
