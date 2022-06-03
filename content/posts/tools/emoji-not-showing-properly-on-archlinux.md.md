---
title: "Emoji Not Showing Properly on Archlinux.md"
date: 2022-06-02T19:26:39-07:00
draft: false
---

The issue persisted for a long time. 

Emojis in kconsole are displayed as blocks(cubes) even though the noto-font-emoji had been installed. Luckily I found a solution today <!--more-->

[Original Solution](https://flammie.github.io/dotfiles/fontconfig.html)

Basically, I inserted the following lines to `/etc/font/fonts.conf` inside the `<font-config>` tag

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