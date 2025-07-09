---
title: Emoji Not Showing Properly on Archlinux.md
categories:
  - System
date: 2022-06-03 02:26:39
---

The issue persisted for a long time. 

Emojis in kconsole are displayed as blocks(cubes) even though the noto-font-emoji had been installed. I found a solution today 

[Original Solution](https://flammie.github.io/dotfiles/fontconfig.html)

Insert the following lines to `/etc/font/fonts.conf` inside the `<font-config>` tag

```xml
<match target="font">
		<test name="family" compare="contains">
			<string>Emoji</string>
		</test>
		<edit name="hinting" mode="assign">
			<bool>true</bool>
		</edit>
		<edit name="hintstyle" mode="assign">
			<const>hintslight</const>
		</edit>
		<edit name="embeddedbitmap" mode="assign">
			<bool>true</bool>
		</edit>
	</match>
```
