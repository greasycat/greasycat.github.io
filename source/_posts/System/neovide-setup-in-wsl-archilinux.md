---
title: Setup Neovide in WSL Archlinux
categories:
  - System
date: 2025-05-01 20:41:00
tags: 
- archlinux
- neovim
- wsl
- neovide
---

# Reading time
- less than 5 min

# Background
Neovide is a one of the best GUI for Neovim. It is written in Rust, supports high refreshing rate, smooth animations and transparent/opacity background. I tried to use Neovide as my daily editor but later walked back to VSCode for some Neovim configuration related reasons. Recently the weather is getting warmer, so is my laptop when a VSCode instance is running with minimal typing. I decided to give Neovide and Neovim another visit and this time I will be running Neovim in WSL.

# Use Neovim RPC

The first bump quickly came as I tried to run neovide with `neovide --wsl`, I have 3 instances of distributions (Ubuntu, Archlinux & NixOS) and have Neovim installed on first 2 distros. But the neovide was not able to detect the Neovim installation. As a workround, I run the following command to use the Neovim [remote](https://Neovim.io/doc/user/remote.html) mode.

```bash
nvim --headless --lisen localhost:6666 
```

then open a powershell in windows, we connect neovide to this address

```powershell
neovide --server=localhost:6666
```

The 1st line will run a Neovim instance on current directory and it is a blocking foreground process and 2nd line will run the neovide and connect to the instance. This is annoying since we have to do 2 things on 2 shells.
A better solution will be using a bash function to handle both jobs:

```bash
function nvs {
  nvim --headless --listen localhost:6666 "$@" &
  sleep 0.5 && "/mnt/c/Program Files/Neovide/neovide.exe" --server=localhost:6666 &
}
```

> You can replace the port and neovide installation path above with your own

By adding the above function to `.bashrc`, and source it. We now have a single command to run a headless Neovim and the corresponding Neovide.

```bash
nvs something.txt
```

# Clipboard
Checkout the clipboard post on how to setup the clipboard for neovim in archlinux
