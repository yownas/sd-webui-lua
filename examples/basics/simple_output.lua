-- Simple output examples - Yownas

-- Clear Output box
ui.clear()

-- Add text to Output
ui.out("Adding text to Output with ui.out()")

-- These two have basically the same functionality
print("Using print()")
ui.console("Using ui.consol()")

-- Log 
ui.log.info("Some information.")
ui.log.warning("A warning!")
ui.log.error("An error! Help!")

-- An optional return() will write to the Output box
a = 1
b = 2
return(a + b)
