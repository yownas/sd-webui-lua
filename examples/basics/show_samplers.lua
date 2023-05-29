-- Simple example on how to list available samplers - Yownas
 
-- To iterate over the list from python we'll use python.iter()

samplers = sd.getsamplers()

ui.out("Samplers:")
for name in python.iter(samplers) do
  ui.out(name)
end
