// @ts-check
import { defineConfig } from 'astro/config';
import remarkToc from 'remark-toc'
import remarkMath from 'remark-math';
import rehypeMathjax from 'rehype-mathjax';

import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://greasycat.github.io',

  markdown: {
    remarkPlugins: [remarkMath, remarkToc],
    rehypePlugins: [rehypeMathjax],
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
    },
  },

  integrations: [sitemap()],
});