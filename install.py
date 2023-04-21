import launch

if not launch.is_installed("lupa"):
    launch.run_pip("install lupa", "requirements for Lua")
