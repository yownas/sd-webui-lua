-- This is the example in the README - Yownas

for i = 1, 12, 1 do
  ui.status("Generating " .. input .. " #" .. i)
  img = sd.process('a cute ' .. input)
  ui.gallery.add(img)
end

return("Done...")
