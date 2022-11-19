---
title: Missing or Oversized Zoom Window on Arch
tags:
  - archlinux
  - zoom
  - solved
draft: false
categories:
  - System
date: 2022-06-01 23:32:52
---

The zoom client installed from AUR (and also the one from flatpak) had two weird issues:

1. the window didn't show up
2. the window is oversize and cannot be rescaled.


checked the terminal output (there's no output) and the log file under `$HOME/.zoom/log` (nothing suspicious).

# Solution
After few google searches, I found a workaround

```sh
vim $HOME/.config/zoomus.conf
```

```toml
autoScale=false
```

I reinstalled the zoom (restart also works) and it worked

