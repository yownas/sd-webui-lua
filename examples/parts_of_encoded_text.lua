-- Copy parts of the Clip encoded text - Yownas
prompt = [[portrait of a cyberpunk assassin, blade runner style, cyberpunk style, blade runner style, sci-fi character,
dark environment. highly detailed painting, 8k, mid shot. cinematic lighting. futurism dystopian setting.
realistic proportions and composition. oil on canvas. award winning. realistic proportions and faces.
dramatic scene, cute sparkles and rainbows]]

-- We want garbage collecting running
collectgarbage(restart)
p = sd.getp()
p.seed = -1
c= sd.textencode(prompt)
uc = sd.negcond('nsfw, out of focus, blurry, boring and other things we do not want to see')

-- Get a completely empty "text encode"
e = torch.new_zeros({1, 77, 768})

-- Copy parts of c into e and generate images along the way
for y = 0, 767, 1 do
  for x = 0, 76, 1 do
    e[0][x][y] = c[0][x][y]
  end
  ui.status("Generating step: " .. tostring(y) .. "/768")
  -- Generate an image every 50th row
  if y%50 == 0 then
    ui.gallery.addc(sd.toimage(sd.vae(sd.sample(p, e, uc))), tostring(math.floor(100*y/767)) .. "%")
  end
end
ui.gallery.addc(sd.toimage(sd.vae(sd.sample(p, c, uc))), "100%")
