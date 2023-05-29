-- Create random prompts - Yownas
adjective = {'cute', 'big', 'small', 'cool'}
animals = {'dog', 'cat', 'mouse', 'rabbit'}
verb = {'sitting', 'sleeping', 'running', 'playing'}

function rnd (table)
  return(table[math.random(1,#table)])
end

for i = 1, 4, 1
do
  prompt = "A " .. rnd(adjective) .. " " .. rnd(animals) .. " " .. rnd(verb)
  img = sd.process(prompt)
  ui.gallery.addc(img, prompt)
end
