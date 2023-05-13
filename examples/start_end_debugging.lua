-- Example to show how to execute only part of a script - Yownas

ui.console("This will not happen.")

--START--

-- Adding a START comment like this will skip execution of the script above it.

ui.console("Debug here!")

--END--

-- Adding a END comment like this will stop execution.

ui.console("This will not happen either.")
