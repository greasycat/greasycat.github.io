---
title: "Missing or Oversized Zoom Window on Arch"
date: 2022-06-01T16:32:52-07:00
tags: [archlinux, zoom, solved]
draft: false
---

# Problem
The zoom client installed from AUR (and also the one from flatpak) had two weird issues:

1. the window didn't show up
2. the window is oversize and cannot be rescaled.<!--more-->


Immediately, I checked the terminal output (there's no output) and the log file under `$HOME/.zoom/log` (nothing suspicious).

# Solution
After few google searches, I found a workaround

```sh
vim $HOME/.config/zoomus.conf
```

```toml
autoScale=false
```

I reinstalled the zoom (restart also works) and it worked
