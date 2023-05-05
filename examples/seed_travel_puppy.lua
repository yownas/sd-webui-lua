-- Seed travel, making a gif using a table of images - Yownas
-- Get a Processing-object
p = sd.getp()

p.prompt = 'a cute puppy'
p.seed = 62345
p.subseed = 3455
p.steps = 10

gif = {}

-- Change subseed_strength from 0 to 1
for s = 0,1,0.1
do
  p.subseed_strength = s
  gif[#gif + 1] = sd.process(p)
end

anim= sd.makegif(gif, 500)

ui.gallery.addc(anim, 'A cute puppy gif')
